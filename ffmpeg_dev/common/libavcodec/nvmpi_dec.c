#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "dynlink_nvmpi.h"
#include "avcodec.h"
#include "decode.h"
#include "internal.h"
#include "libavutil/buffer.h"
#include "libavutil/common.h"
#include "libavutil/frame.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_drm.h"
#include "libavutil/imgutils.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#if LIBAVCODEC_VERSION_MAJOR >= 60
#include "codec_internal.h"
#endif

#include <libdrm/drm_fourcc.h>

#define OPT_frame_pool_size_MIN 1
#define OPT_frame_pool_size_MAX 32
#define OPT_frame_pool_size_DEFAULT 10

typedef struct {
	char eos_reached;
	nvmpictx* ctx;
	AVClass *av_class;
	AVFrame *bufFrame;
	char *resize_expr;
	int frame_pool_size;
	int drm_prime;  /* 1 = output DRM_PRIME frames (zero-copy) */
	int p010;       /* 1 = request P010 (10-bit) frame pool from nvmpi */
	int is_10bit;   /* Set after first frame: decoder detected 10-bit content */

	/* DRM hardware context for DRM_PRIME mode */
	AVBufferRef *device_ref;
	AVBufferRef *frames_ref;
} nvmpiDecodeContext;

/* Release callback for DRM_PRIME frames — returns buffer to decoder pool */
typedef struct {
	nvmpi_frame_release_cb release;
	void *opaque;
} nvmpiDRMFrameRef;

static void drm_frame_free(void *opaque, uint8_t *data)
{
	nvmpiDRMFrameRef *ref = (nvmpiDRMFrameRef *)opaque;
	if (ref->release)
		ref->release(ref->opaque);
	av_free(data); /* free the AVDRMFrameDescriptor */
	av_free(ref);
}

static nvCodingType nvmpi_get_codingtype(AVCodecContext *avctx)
{
	switch (avctx->codec_id) {
		case AV_CODEC_ID_H264:          return NV_VIDEO_CodingH264;
		case AV_CODEC_ID_HEVC:          return NV_VIDEO_CodingHEVC;
		case AV_CODEC_ID_VP8:           return NV_VIDEO_CodingVP8;
		case AV_CODEC_ID_VP9:           return NV_VIDEO_CodingVP9;
		case AV_CODEC_ID_MPEG4:		return NV_VIDEO_CodingMPEG4;
		case AV_CODEC_ID_MPEG2VIDEO:    return NV_VIDEO_CodingMPEG2;
		default:                        return NV_VIDEO_CodingUnused;
	}
};


static int nvmpi_init_decoder(AVCodecContext *avctx)
{
	nvmpiDecodeContext *nvmpi_context = avctx->priv_data;
	nvDecParam param={0};
	
	param.codingType =nvmpi_get_codingtype(avctx);
	if (param.codingType == NV_VIDEO_CodingUnused)
	{
		av_log(avctx, AV_LOG_ERROR, "Unknown codec type (%d).\n", avctx->codec_id);
		return AVERROR_UNKNOWN;
	}
	
	param.frame_pool_size = nvmpi_context->frame_pool_size;
	if(param.frame_pool_size < OPT_frame_pool_size_MIN || param.frame_pool_size > OPT_frame_pool_size_MAX)
	{
		av_log(avctx, AV_LOG_WARNING, "Incorrect frame_pool_size specified: %d. Default (%d) will be used.\n", param.frame_pool_size, OPT_frame_pool_size_DEFAULT);
		param.frame_pool_size = OPT_frame_pool_size_DEFAULT;
	}

	/* P010 option: request 10-bit frame pool from nvmpi */
	if (nvmpi_context->p010) {
		param.pixFormat = NV_PIX_P010;
		nvmpi_context->is_10bit = 1;
		avctx->pix_fmt = AV_PIX_FMT_P010LE;
	} else {
		param.pixFormat = NV_PIX_YUV420;
		/* Default pix_fmt workaround */
		if (avctx->pix_fmt == AV_PIX_FMT_NONE)
			avctx->pix_fmt = AV_PIX_FMT_YUV420P;
		else if (avctx->pix_fmt != AV_PIX_FMT_YUV420P &&
		         avctx->pix_fmt != AV_PIX_FMT_YUVJ420P) {
			/* Non-P010 mode only supports 8-bit formats */
			av_log(avctx, AV_LOG_WARNING, "Unsupported pix_fmt for NVMPI, defaulting to YUV420P\n");
			avctx->pix_fmt = AV_PIX_FMT_YUV420P;
		}
	}

    if (nvmpi_context->resize_expr && sscanf(nvmpi_context->resize_expr, "%dx%d",
                                             &param.resized.width, &param.resized.height) != 2)
	{
        av_log(avctx, AV_LOG_ERROR, "Invalid resize expressions\n");
        return AVERROR(EINVAL);
    }
	
	//overwrite avctx w and h if resize option is used
	if(param.resized.width && param.resized.height)
	{
		avctx->width = param.resized.width;
		avctx->height = param.resized.height;
	}

	if (nvmpi_dynlink_load() < 0) {
		av_log(avctx, AV_LOG_ERROR, "Failed to load libnvmpi.so: %s\n",
		       dlerror());
		return AVERROR_EXTERNAL;
	}

	if (nvmpi_context->drm_prime) {
		/* DRM_PRIME mode: set up hardware contexts, no CPU buffer needed */
		AVHWDeviceContext *device_ctx;
		AVDRMDeviceContext *drm_ctx;
		int ret;

		nvmpi_context->device_ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_DRM);
		if (!nvmpi_context->device_ref)
			return AVERROR(ENOMEM);
		device_ctx = (AVHWDeviceContext *)nvmpi_context->device_ref->data;
		drm_ctx = device_ctx->hwctx;
		drm_ctx->fd = -1; /* No DRM render node needed */
		ret = av_hwdevice_ctx_init(nvmpi_context->device_ref);
		if (ret < 0) {
			av_buffer_unref(&nvmpi_context->device_ref);
			return ret;
		}

		avctx->pix_fmt = AV_PIX_FMT_DRM_PRIME;
		avctx->sw_pix_fmt = nvmpi_context->is_10bit ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
	} else {
		/* CPU mode: allocate buffer frame */
		nvmpi_context->bufFrame = av_frame_alloc();
		nvmpi_context->bufFrame->width = avctx->width;
		nvmpi_context->bufFrame->height = avctx->height;
		if (ff_get_buffer(avctx, nvmpi_context->bufFrame, 0) < 0)
		{
			av_frame_free(&(nvmpi_context->bufFrame));
			nvmpi_context->bufFrame = NULL;
			return AVERROR(ENOMEM);
		}
	}

	nvmpi_context->ctx=nvmpi_create_decoder(&param);

	if(!nvmpi_context->ctx)
	{
		if (nvmpi_context->bufFrame) {
			av_frame_free(&(nvmpi_context->bufFrame));
			nvmpi_context->bufFrame = NULL;
		}
		av_buffer_unref(&nvmpi_context->device_ref);
		av_log(avctx, AV_LOG_ERROR, "Failed to nvmpi_create_decoder (code = %d).\n", AVERROR_EXTERNAL);
		return AVERROR_EXTERNAL;
	}

   return 0;
}

static int nvmpi_close(AVCodecContext *avctx)
{
	nvmpiDecodeContext *nvmpi_context = avctx->priv_data;
	if(nvmpi_context->bufFrame)
	{
		av_frame_free(&(nvmpi_context->bufFrame));
		nvmpi_context->bufFrame = NULL;
	}
	av_buffer_unref(&nvmpi_context->frames_ref);
	av_buffer_unref(&nvmpi_context->device_ref);
	return nvmpi_decoder_close(nvmpi_context->ctx);
}

#if LIBAVCODEC_VERSION_MAJOR >= 60
static int nvmpi_decode(AVCodecContext *avctx, AVFrame *data, int *got_frame, AVPacket *avpkt)
#else
static int nvmpi_decode(AVCodecContext *avctx, void *data, int *got_frame, AVPacket *avpkt)
#endif
{
	nvmpiDecodeContext *nvmpi_context = avctx->priv_data;
	AVFrame *frame = data;
	nvPacket packet;
	int res;
	int decode_ret = avpkt->size;

	if(avpkt->size)
	{
		packet.payload_size=avpkt->size;
		packet.payload=avpkt->data;
		packet.pts=avpkt->pts;

		res=nvmpi_decoder_put_packet(nvmpi_context->ctx,&packet);
		if(res < 0)
		{
			if(res == -1)
			{
				decode_ret = AVERROR(EAGAIN); //TODO log
			}
			//TODO error handling
		}
	}

	if (nvmpi_context->drm_prime) {
		/* DRM_PRIME path: get DMA-BUF fd, wrap in AVDRMFrameDescriptor */
		int fd, w, h, pitch;
		int64_t timestamp;
		nvmpi_frame_release_cb release;
		void *opaque;
		AVDRMFrameDescriptor *desc;
		nvmpiDRMFrameRef *ref;

		res = nvmpi_decoder_get_frame_fd(nvmpi_context->ctx, &fd, &w, &h,
		                                  &pitch, &timestamp, &release, &opaque);
		if (res < 0)
			return decode_ret;

		/* Check bit depth from nvmpi (valid after first frame decoded) */
		if (!nvmpi_context->is_10bit) {
			int bd = nvmpi_decoder_get_bit_depth(nvmpi_context->ctx);
			if (bd == 10)
				nvmpi_context->is_10bit = 1;
		}

		/* Create frames context lazily on first frame (we now know dimensions) */
		if (!nvmpi_context->frames_ref) {
			AVHWFramesContext *frames_ctx;
			nvmpi_context->frames_ref = av_hwframe_ctx_alloc(nvmpi_context->device_ref);
			if (!nvmpi_context->frames_ref) {
				release(opaque);
				return AVERROR(ENOMEM);
			}
			frames_ctx = (AVHWFramesContext *)nvmpi_context->frames_ref->data;
			frames_ctx->format    = AV_PIX_FMT_DRM_PRIME;
			frames_ctx->sw_format = nvmpi_context->is_10bit ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
			frames_ctx->width     = w;
			frames_ctx->height    = h;
			res = av_hwframe_ctx_init(nvmpi_context->frames_ref);
			if (res < 0) {
				av_buffer_unref(&nvmpi_context->frames_ref);
				release(opaque);
				return res;
			}
			/* Update avctx dimensions from actual decoded size */
			avctx->width = w;
			avctx->height = h;
		}

		desc = av_mallocz(sizeof(*desc));
		if (!desc) {
			release(opaque);
			return AVERROR(ENOMEM);
		}

		desc->nb_objects = 1;
		desc->objects[0].fd   = fd;
		desc->objects[0].size = pitch * h * 3 / 2; /* NV12/P010: Y + UV (same layout, P010 has wider pitch) */

		desc->nb_layers = 1;
		desc->layers[0].format    = nvmpi_context->is_10bit ? DRM_FORMAT_P010 : DRM_FORMAT_NV12;
		desc->layers[0].nb_planes = 2;
		/* Y plane */
		desc->layers[0].planes[0].object_index = 0;
		desc->layers[0].planes[0].offset       = 0;
		desc->layers[0].planes[0].pitch        = pitch;
		/* UV plane */
		desc->layers[0].planes[1].object_index = 0;
		desc->layers[0].planes[1].offset       = pitch * h;
		desc->layers[0].planes[1].pitch        = pitch;

		/* Wrap release callback so pool buffer is returned when AVFrame is freed */
		ref = av_mallocz(sizeof(*ref));
		if (!ref) {
			av_free(desc);
			release(opaque);
			return AVERROR(ENOMEM);
		}
		ref->release = release;
		ref->opaque  = opaque;

		frame->data[0] = (uint8_t *)desc;
		frame->format  = AV_PIX_FMT_DRM_PRIME;
		frame->width   = w;
		frame->height  = h;
		frame->hw_frames_ctx = av_buffer_ref(nvmpi_context->frames_ref);
		frame->buf[0]  = av_buffer_create((uint8_t *)desc, sizeof(*desc),
		                                   drm_frame_free, ref,
		                                   AV_BUFFER_FLAG_READONLY);
		frame->pts     = timestamp;
		frame->pkt_dts = AV_NOPTS_VALUE;

		*got_frame = 1;
	} else {
		/* CPU path: existing memcpy-based decode */
		AVFrame *bufFrame = nvmpi_context->bufFrame;
		nvFrame _nvframe={0};

		_nvframe.payload[0] = bufFrame->data[0];
		_nvframe.payload[1] = bufFrame->data[1];
		_nvframe.payload[2] = bufFrame->data[2];
		_nvframe.linesize[0] = bufFrame->linesize[0];
		_nvframe.linesize[1] = bufFrame->linesize[1];
		_nvframe.linesize[2] = bufFrame->linesize[2];

		res=nvmpi_decoder_get_frame(nvmpi_context->ctx,&_nvframe,avctx->flags & AV_CODEC_FLAG_LOW_DELAY);

		if(res<0)
		{
			return decode_ret;
		}

		/* Check bit depth on first successful frame */
		if (!nvmpi_context->is_10bit) {
			int bd = nvmpi_decoder_get_bit_depth(nvmpi_context->ctx);
			if (bd == 10)
				nvmpi_context->is_10bit = 1;
		}

		bufFrame->format = nvmpi_context->is_10bit ? AV_PIX_FMT_P010LE : AV_PIX_FMT_YUV420P;
		bufFrame->pts=_nvframe.timestamp;
		bufFrame->pkt_dts = AV_NOPTS_VALUE;
		av_frame_move_ref(frame, bufFrame);

		*got_frame = 1;

		bufFrame->width = avctx->width;
		bufFrame->height = avctx->height;
		if (nvmpi_context->is_10bit)
			avctx->pix_fmt = AV_PIX_FMT_P010LE;
		if (ff_get_buffer(avctx, bufFrame, 0) < 0)
		{
			av_log(avctx, AV_LOG_ERROR, "ff_get_buffer failed\n");
			return AVERROR(ENOMEM);
		}

		frame->metadata = bufFrame->metadata;
		bufFrame->metadata = NULL;
	}

	return decode_ret;
}



#define OFFSET(x) offsetof(nvmpiDecodeContext, x)
#define VD AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM
static const AVOption options[] = {
    { "resize",   "Resize (width)x(height)", OFFSET(resize_expr), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VD, "resize" },
    { "frame_pool_size", "Number of frames that could be buffered in the decoder before user must read it with avcodec_receive_frame()", OFFSET(frame_pool_size), AV_OPT_TYPE_INT, {.i64 = OPT_frame_pool_size_DEFAULT }, OPT_frame_pool_size_MIN, OPT_frame_pool_size_MAX, VD, "frame_pool_size" },
    { "drm_prime", "Output DRM_PRIME (DMA-BUF) frames for zero-copy pipeline", OFFSET(drm_prime), AV_OPT_TYPE_BOOL, {.i64 = 0}, 0, 1, VD },
    { "p010", "Use P010 (10-bit) frame pool for 10-bit content", OFFSET(p010), AV_OPT_TYPE_BOOL, {.i64 = 0}, 0, 1, VD },
    { NULL }
};

#define NVMPI_DEC_CLASS(NAME) \
	static const AVClass nvmpi_##NAME##_dec_class = { \
		.class_name = "nvmpi_" #NAME "_dec", \
		.option     = options, \
		.version    = LIBAVUTIL_VERSION_INT, \
	};

#if LIBAVCODEC_VERSION_MAJOR >= 60
	#define NVMPI_DEC(NAME, ID, BSFS) \
		NVMPI_DEC_CLASS(NAME) \
		FFCodec ff_##NAME##_nvmpi_decoder = { \
			.p.name           = #NAME "_nvmpi", \
			CODEC_LONG_NAME(#NAME " (nvmpi)"), \
			.p.type           = AVMEDIA_TYPE_VIDEO, \
			.p.id             = ID, \
			.priv_data_size = sizeof(nvmpiDecodeContext), \
			.init           = nvmpi_init_decoder, \
			.close          = nvmpi_close, \
			FF_CODEC_DECODE_CB(nvmpi_decode), \
			.p.priv_class     = &nvmpi_##NAME##_dec_class, \
			.p.capabilities   = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_AVOID_PROBING | AV_CODEC_CAP_HARDWARE, \
			.p.pix_fmts	=(const enum AVPixelFormat[]){AV_PIX_FMT_DRM_PRIME,AV_PIX_FMT_YUV420P,AV_PIX_FMT_NV12,AV_PIX_FMT_NONE},\
			.bsfs           = BSFS, \
			.p.wrapper_name   = "nvmpi", \
		};
#else
	#define NVMPI_DEC(NAME, ID, BSFS) \
		NVMPI_DEC_CLASS(NAME) \
		AVCodec ff_##NAME##_nvmpi_decoder = { \
			.name           = #NAME "_nvmpi", \
			.long_name      = NULL_IF_CONFIG_SMALL(#NAME " (nvmpi)"), \
			.type           = AVMEDIA_TYPE_VIDEO, \
			.id             = ID, \
			.priv_data_size = sizeof(nvmpiDecodeContext), \
			.init           = nvmpi_init_decoder, \
			.close          = nvmpi_close, \
			.decode         = nvmpi_decode, \
			.priv_class     = &nvmpi_##NAME##_dec_class, \
			.capabilities   = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_AVOID_PROBING | AV_CODEC_CAP_HARDWARE, \
			.pix_fmts	=(const enum AVPixelFormat[]){AV_PIX_FMT_DRM_PRIME,AV_PIX_FMT_YUV420P,AV_PIX_FMT_NV12,AV_PIX_FMT_NONE},\
			.bsfs           = BSFS, \
			.wrapper_name   = "nvmpi", \
		};
#endif


NVMPI_DEC(h264,  AV_CODEC_ID_H264,"h264_mp4toannexb");
NVMPI_DEC(hevc,  AV_CODEC_ID_HEVC,"hevc_mp4toannexb");
NVMPI_DEC(mpeg2, AV_CODEC_ID_MPEG2VIDEO,NULL);
NVMPI_DEC(mpeg4, AV_CODEC_ID_MPEG4,NULL);
NVMPI_DEC(vp9,  AV_CODEC_ID_VP9,NULL);
NVMPI_DEC(vp8, AV_CODEC_ID_VP8,NULL);

