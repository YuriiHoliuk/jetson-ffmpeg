/*
 * vf_scale_vic.c — Hardware scaling using NVIDIA Jetson VIC engine
 *
 * Accepts DRM_PRIME (DMA-BUF) frames, scales via NvBufSurfTransform (VIC),
 * outputs new DRM_PRIME frames at target resolution.
 *
 * Uses dlopen for libnvbufsurface.so to avoid build-time dependency.
 */

#include <dlfcn.h>
#include <stdio.h>

#include "libavutil/avstring.h"
#include "libavutil/eval.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_drm.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"

#include <libdrm/drm_fourcc.h>

/* ---- NvBufSurface types (from nvbufsurface.h) ---- */

typedef struct _NvBufSurfacePlaneParams {
    unsigned int num_planes;
    unsigned int width[4];
    unsigned int height[4];
    unsigned int pitch[4];
    unsigned int offset[4];
    unsigned int psize[4];
    unsigned int bytesPerPix[4];
} NvBufSurfacePlaneParams;

typedef struct _NvBufSurfaceParams {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    int colorFormat;
    int layout;
    int bufferDesc;          /* DMA-BUF fd */
    unsigned int dataSize;
    void *dataPtr;
    NvBufSurfacePlaneParams planeParams;
    /* ... more fields we don't need */
} NvBufSurfaceParams;

typedef struct _NvBufSurface {
    int gpuId;
    unsigned int batchSize;
    unsigned int numFilled;
    int isContiguous;
    int memType;
    NvBufSurfaceParams *surfaceList;
} NvBufSurface;

typedef struct _NvBufSurfaceCreateParams {
    int gpuId;
    unsigned int width;
    unsigned int height;
    unsigned int size;
    int isContiguous;
    int colorFormat;
    int layout;
    int memType;
} NvBufSurfaceCreateParams;

typedef struct _NvBufSurfTransformRect {
    unsigned int top;
    unsigned int left;
    unsigned int width;
    unsigned int height;
} NvBufSurfTransformRect;

typedef struct _NvBufSurfTransformParams {
    unsigned int transform_flag;
    int transform_flip;
    int transform_filter;
    NvBufSurfTransformRect *src_rect;
    NvBufSurfTransformRect *dst_rect;
} NvBufSurfTransformParams;

typedef struct _NvBufSurfTransformConfigParams {
    int compute_mode;
    int gpu_id;
    void *cuda_stream;
} NvBufSurfTransformConfigParams;

/* Enums */
#define NVBUF_COLOR_FORMAT_NV12 4
#define NVBUF_LAYOUT_PITCH 1
#define NVBUF_MEM_SURFACE_ARRAY 4
#define NVBUFSURF_TRANSFORM_FILTER (1 << 1)
#define NvBufSurfTransformCompute_VIC 2
#define NvBufSurfTransform_Inter_Bilinear 1

/* ---- dlopen function pointers ---- */

typedef int (*pf_NvBufSurfaceCreate)(NvBufSurface **surf, int batchSize,
                                      NvBufSurfaceCreateParams *params);
typedef int (*pf_NvBufSurfaceDestroy)(NvBufSurface *surf);
typedef int (*pf_NvBufSurfaceFromFd)(int dmabuf_fd, void **buffer);
typedef int (*pf_NvBufSurfTransform)(NvBufSurface *src, NvBufSurface *dst,
                                      NvBufSurfTransformParams *params);
typedef int (*pf_NvBufSurfTransformSetSessionParams)(
                                      NvBufSurfTransformConfigParams *params);

static pf_NvBufSurfaceCreate         dl_NvBufSurfaceCreate;
static pf_NvBufSurfaceDestroy        dl_NvBufSurfaceDestroy;
static pf_NvBufSurfaceFromFd         dl_NvBufSurfaceFromFd;
static pf_NvBufSurfTransform         dl_NvBufSurfTransform;
static pf_NvBufSurfTransformSetSessionParams dl_NvBufSurfTransformSetSessionParams;

static void *nvbufsurface_lib;

static int load_nvbufsurface(void *log_ctx)
{
    if (nvbufsurface_lib)
        return 0;

    nvbufsurface_lib = dlopen("libnvbufsurface.so", RTLD_LAZY);
    if (!nvbufsurface_lib) {
        av_log(log_ctx, AV_LOG_ERROR, "Failed to load libnvbufsurface.so: %s\n",
               dlerror());
        return AVERROR_EXTERNAL;
    }

#define LOAD(name) do { \
    dl_##name = (pf_##name)dlsym(nvbufsurface_lib, #name); \
    if (!dl_##name) { \
        av_log(log_ctx, AV_LOG_ERROR, "Missing symbol: %s\n", #name); \
        dlclose(nvbufsurface_lib); \
        nvbufsurface_lib = NULL; \
        return AVERROR_EXTERNAL; \
    } \
} while (0)

    LOAD(NvBufSurfaceCreate);
    LOAD(NvBufSurfaceDestroy);
    LOAD(NvBufSurfaceFromFd);
    LOAD(NvBufSurfTransform);
    LOAD(NvBufSurfTransformSetSessionParams);

#undef LOAD
    return 0;
}

/* ---- Filter context ---- */

typedef struct ScaleVICContext {
    const AVClass *class;
    char *w_expr;
    char *h_expr;
    int out_w, out_h;

    AVBufferRef *device_ref;
    AVBufferRef *frames_ref;

    NvBufSurfTransformConfigParams session;
    int session_initialized;
} ScaleVICContext;

/* ---- Release callback for output frames ---- */

static void vic_frame_free(void *opaque, uint8_t *data)
{
    NvBufSurface *surf = (NvBufSurface *)opaque;
    if (surf)
        dl_NvBufSurfaceDestroy(surf);
    av_free(data);
}

/* ---- Filter implementation ---- */

static av_cold int scale_vic_init(AVFilterContext *avctx)
{
    int ret = load_nvbufsurface(avctx);
    if (ret < 0)
        return ret;
    return 0;
}

static av_cold void scale_vic_uninit(AVFilterContext *avctx)
{
    ScaleVICContext *ctx = avctx->priv;
    av_buffer_unref(&ctx->frames_ref);
    av_buffer_unref(&ctx->device_ref);
}

static int scale_vic_config_output(AVFilterLink *outlink)
{
    AVFilterContext *avctx = outlink->src;
    ScaleVICContext *ctx = avctx->priv;
    AVFilterLink *inlink = avctx->inputs[0];
    AVHWFramesContext *frames_ctx;
    AVDRMDeviceContext *drm_ctx;
    AVHWDeviceContext *device_ctx;
    double var_values[] = { inlink->w, inlink->h, 0 };
    static const char *const var_names[] = { "in_w", "in_h", NULL };
    double res;
    int ret;

    /* Resolve width expression */
    if (ctx->w_expr) {
        ret = av_expr_parse_and_eval(&res, ctx->w_expr, var_names, var_values,
                                     NULL, NULL, NULL, NULL, NULL, 0, avctx);
        if (ret < 0) return ret;
        ctx->out_w = (int)res;
    }
    if (ctx->out_w <= 0) ctx->out_w = inlink->w;

    /* Resolve height expression */
    if (ctx->h_expr) {
        var_values[2] = ctx->out_w; /* allow referencing out_w in h_expr */
        ret = av_expr_parse_and_eval(&res, ctx->h_expr, var_names, var_values,
                                     NULL, NULL, NULL, NULL, NULL, 0, avctx);
        if (ret < 0) return ret;
        ctx->out_h = (int)res;
    }
    if (ctx->out_h <= 0) ctx->out_h = inlink->h;

    /* Handle -1 (auto aspect ratio) */
    if (ctx->out_w == -1)
        ctx->out_w = av_rescale(ctx->out_h, inlink->w, inlink->h) & ~1;
    if (ctx->out_h == -1)
        ctx->out_h = av_rescale(ctx->out_w, inlink->h, inlink->w) & ~1;

    /* Ensure even dimensions */
    ctx->out_w = (ctx->out_w + 1) & ~1;
    ctx->out_h = (ctx->out_h + 1) & ~1;

    outlink->w = ctx->out_w;
    outlink->h = ctx->out_h;

    /* Preserve sample aspect ratio */
    if (inlink->sample_aspect_ratio.num)
        outlink->sample_aspect_ratio = av_mul_q(
            (AVRational){outlink->h * inlink->w, outlink->w * inlink->h},
            inlink->sample_aspect_ratio);
    else
        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;

    /* Create DRM device context */
    ctx->device_ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_DRM);
    if (!ctx->device_ref)
        return AVERROR(ENOMEM);
    device_ctx = (AVHWDeviceContext *)ctx->device_ref->data;
    drm_ctx = device_ctx->hwctx;
    drm_ctx->fd = -1;
    ret = av_hwdevice_ctx_init(ctx->device_ref);
    if (ret < 0) return ret;

    /* Create output HW frames context */
    ctx->frames_ref = av_hwframe_ctx_alloc(ctx->device_ref);
    if (!ctx->frames_ref)
        return AVERROR(ENOMEM);
    frames_ctx = (AVHWFramesContext *)ctx->frames_ref->data;
    frames_ctx->format    = AV_PIX_FMT_DRM_PRIME;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width     = ctx->out_w;
    frames_ctx->height    = ctx->out_h;
    ret = av_hwframe_ctx_init(ctx->frames_ref);
    if (ret < 0) return ret;

    outlink->hw_frames_ctx = av_buffer_ref(ctx->frames_ref);

    av_log(avctx, AV_LOG_VERBOSE, "scale_vic: %dx%d -> %dx%d\n",
           inlink->w, inlink->h, ctx->out_w, ctx->out_h);

    return 0;
}

static int scale_vic_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *avctx = inlink->dst;
    ScaleVICContext *ctx = avctx->priv;
    AVFilterLink *outlink = avctx->outputs[0];
    AVDRMFrameDescriptor *in_desc;
    AVDRMFrameDescriptor *out_desc;
    NvBufSurface *in_surf = NULL;
    NvBufSurface *out_surf = NULL;
    NvBufSurfaceCreateParams create_params;
    NvBufSurfTransformParams xform;
    AVFrame *out;
    int in_fd, out_fd, out_pitch;
    int ret;

    /* Initialize VIC session on first frame */
    if (!ctx->session_initialized) {
        memset(&ctx->session, 0, sizeof(ctx->session));
        ctx->session.compute_mode = NvBufSurfTransformCompute_VIC;
        ctx->session.gpu_id = 0;
        dl_NvBufSurfTransformSetSessionParams(&ctx->session);
        ctx->session_initialized = 1;
    }

    /* Extract input DMA-BUF fd */
    in_desc = (AVDRMFrameDescriptor *)in->data[0];
    in_fd = in_desc->objects[0].fd;

    /* Get input NvBufSurface */
    ret = dl_NvBufSurfaceFromFd(in_fd, (void **)&in_surf);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "NvBufSurfaceFromFd failed for input fd=%d\n", in_fd);
        av_frame_free(&in);
        return AVERROR_EXTERNAL;
    }

    /* Allocate output NvBufSurface */
    memset(&create_params, 0, sizeof(create_params));
    create_params.gpuId       = 0;
    create_params.width       = ctx->out_w;
    create_params.height      = ctx->out_h;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
    create_params.layout      = NVBUF_LAYOUT_PITCH;
    create_params.memType     = NVBUF_MEM_SURFACE_ARRAY;

    ret = dl_NvBufSurfaceCreate(&out_surf, 1, &create_params);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "NvBufSurfaceCreate failed\n");
        av_frame_free(&in);
        return AVERROR_EXTERNAL;
    }

    /* VIC hardware scaling */
    memset(&xform, 0, sizeof(xform));
    xform.transform_flag   = NVBUFSURF_TRANSFORM_FILTER;
    xform.transform_filter = NvBufSurfTransform_Inter_Bilinear;

    ret = dl_NvBufSurfTransform(in_surf, out_surf, &xform);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "NvBufSurfTransform failed\n");
        dl_NvBufSurfaceDestroy(out_surf);
        av_frame_free(&in);
        return AVERROR_EXTERNAL;
    }

    /* Wrap output in AVDRMFrameDescriptor */
    out_fd    = out_surf->surfaceList[0].bufferDesc;
    out_pitch = out_surf->surfaceList[0].planeParams.pitch[0];

    out_desc = av_mallocz(sizeof(*out_desc));
    if (!out_desc) {
        dl_NvBufSurfaceDestroy(out_surf);
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }

    out_desc->nb_objects = 1;
    out_desc->objects[0].fd   = out_fd;
    out_desc->objects[0].size = out_pitch * ctx->out_h * 3 / 2;

    out_desc->nb_layers = 1;
    out_desc->layers[0].format    = DRM_FORMAT_NV12;
    out_desc->layers[0].nb_planes = 2;
    out_desc->layers[0].planes[0].object_index = 0;
    out_desc->layers[0].planes[0].offset       = 0;
    out_desc->layers[0].planes[0].pitch        = out_pitch;
    out_desc->layers[0].planes[1].object_index = 0;
    out_desc->layers[0].planes[1].offset       = out_pitch * ctx->out_h;
    out_desc->layers[0].planes[1].pitch        = out_pitch;

    /* Build output AVFrame */
    out = av_frame_alloc();
    if (!out) {
        av_free(out_desc);
        dl_NvBufSurfaceDestroy(out_surf);
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }

    out->data[0]       = (uint8_t *)out_desc;
    out->format        = AV_PIX_FMT_DRM_PRIME;
    out->width         = ctx->out_w;
    out->height        = ctx->out_h;
    out->hw_frames_ctx = av_buffer_ref(ctx->frames_ref);
    out->buf[0]        = av_buffer_create((uint8_t *)out_desc, sizeof(*out_desc),
                                          vic_frame_free, out_surf,
                                          AV_BUFFER_FLAG_READONLY);
    out->pts           = in->pts;
    out->pkt_dts       = in->pkt_dts;
    out->duration      = in->duration;
    out->sample_aspect_ratio = outlink->sample_aspect_ratio;

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

/* ---- Filter registration ---- */

#define OFFSET(x) offsetof(ScaleVICContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM

static const AVOption scale_vic_options[] = {
    { "w", "Output video width",  OFFSET(w_expr), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS },
    { "h", "Output video height", OFFSET(h_expr), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(scale_vic);

static const AVFilterPad scale_vic_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = scale_vic_filter_frame,
    },
};

static const AVFilterPad scale_vic_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = scale_vic_config_output,
    },
};

const AVFilter ff_vf_scale_vic = {
    .name           = "scale_vic",
    .description    = NULL_IF_CONFIG_SMALL("Scale using NVIDIA Jetson VIC engine"),
    .priv_class     = &scale_vic_class,
    .priv_size      = sizeof(ScaleVICContext),
    .init           = scale_vic_init,
    .uninit         = scale_vic_uninit,
    FILTER_INPUTS(scale_vic_inputs),
    FILTER_OUTPUTS(scale_vic_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_DRM_PRIME),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
