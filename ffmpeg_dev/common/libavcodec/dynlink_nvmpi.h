/*
 * dynlink_nvmpi.h — dlopen/dlsym wrapper for libnvmpi
 *
 * Loads libnvmpi.so lazily at codec init time instead of at ffmpeg startup.
 * This prevents EGL initialization crashes when nvmpi codecs aren't being used
 * (e.g., during ffprobe, codec copy, or listing codecs).
 *
 * Follows the same pattern as FFmpeg's dynlink_loader.h for nv-codec-headers.
 */

#ifndef DYNLINK_NVMPI_H
#define DYNLINK_NVMPI_H

#include <dlfcn.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

/* ---- Struct/enum definitions from nvmpi.h ---- */

#define NVMPI_ENC_CHUNK_SIZE 2*1024*1024

typedef struct nvmpictx nvmpictx;

typedef enum {
    NV_PIX_NV12,
    NV_PIX_YUV420
} nvPixFormat;

typedef enum {
    NV_VIDEO_CodingUnused,
    NV_VIDEO_CodingH264,
    NV_VIDEO_CodingMPEG4,
    NV_VIDEO_CodingMPEG2,
    NV_VIDEO_CodingVP8,
    NV_VIDEO_CodingVP9,
    NV_VIDEO_CodingHEVC,
} nvCodingType;

typedef struct _NVSIZE {
    unsigned int width;
    unsigned int height;
} nvSize;

typedef struct _NVENCPARAM {
    unsigned int width;
    unsigned int height;
    unsigned int profile;
    unsigned int level;
    unsigned int bitrate;
    unsigned int peak_bitrate;
    char enableLossless;
    char mode_vbr;
    char insert_spspps_idr;
    unsigned int iframe_interval;
    unsigned int idr_interval;
    unsigned int fps_n;
    unsigned int fps_d;
    int capture_num;
    unsigned int max_b_frames;
    unsigned int refs;
    unsigned int qmax;
    unsigned int qmin;
    unsigned int hw_preset_type;
    unsigned int vbv_buffer_size;
    nvCodingType codingType;
    int use_dmabuf;
} nvEncParam;

typedef struct _NVDECPARAM {
    int frame_pool_size;
    nvCodingType codingType;
    nvPixFormat pixFormat;
    nvSize resized;
} nvDecParam;

typedef struct _NVPACKET {
    unsigned long flags;
    unsigned long payload_size;
    unsigned char *payload;
    unsigned long pts;
    void *privData;
} nvPacket;

typedef struct _NVFRAME {
    unsigned long flags;
    unsigned long payload_size[3];
    unsigned char *payload[3];
    unsigned int linesize[3];
    nvPixFormat type;
    unsigned int width;
    unsigned int height;
    time_t timestamp;
} nvFrame;

/* ---- Release callback for DMA-BUF frame references ---- */
typedef void (*nvmpi_frame_release_cb)(void *opaque);

/* ---- Function pointer types ---- */

typedef nvmpictx *(*pf_nvmpi_create_decoder)(nvDecParam *param);
typedef int       (*pf_nvmpi_decoder_put_packet)(nvmpictx *ctx, nvPacket *packet);
typedef int       (*pf_nvmpi_decoder_get_frame)(nvmpictx *ctx, nvFrame *frame, bool wait);
typedef int       (*pf_nvmpi_decoder_get_frame_fd)(nvmpictx *ctx, int *dmabuf_fd,
                    int *width, int *height, int *pitch, int64_t *timestamp,
                    nvmpi_frame_release_cb *release, void **opaque);
typedef int       (*pf_nvmpi_decoder_close)(nvmpictx *ctx);
typedef nvmpictx *(*pf_nvmpi_create_encoder)(nvEncParam *param);
typedef int       (*pf_nvmpi_encoder_put_frame)(nvmpictx *ctx, nvFrame *frame);
typedef int       (*pf_nvmpi_encoder_put_frame_fd)(nvmpictx *ctx, int dmabuf_fd,
                    int width, int height, int pitch, int64_t timestamp);
typedef int       (*pf_nvmpi_encoder_get_packet)(nvmpictx *ctx, nvPacket **packet);
typedef int       (*pf_nvmpi_encoder_dqEmptyPacket)(nvmpictx *ctx, nvPacket **packet);
typedef void      (*pf_nvmpi_encoder_qEmptyPacket)(nvmpictx *ctx, nvPacket *packet);
typedef int       (*pf_nvmpi_encoder_close)(nvmpictx *ctx);

/* ---- Global function pointers ---- */

static pf_nvmpi_create_decoder        nvmpi_create_decoder;
static pf_nvmpi_decoder_put_packet    nvmpi_decoder_put_packet;
static pf_nvmpi_decoder_get_frame     nvmpi_decoder_get_frame;
static pf_nvmpi_decoder_get_frame_fd  nvmpi_decoder_get_frame_fd;
static pf_nvmpi_decoder_close         nvmpi_decoder_close;
static pf_nvmpi_create_encoder        nvmpi_create_encoder;
static pf_nvmpi_encoder_put_frame     nvmpi_encoder_put_frame;
static pf_nvmpi_encoder_put_frame_fd  nvmpi_encoder_put_frame_fd;
static pf_nvmpi_encoder_get_packet    nvmpi_encoder_get_packet;
static pf_nvmpi_encoder_dqEmptyPacket nvmpi_encoder_dqEmptyPacket;
static pf_nvmpi_encoder_qEmptyPacket  nvmpi_encoder_qEmptyPacket;
static pf_nvmpi_encoder_close         nvmpi_encoder_close;

static void *nvmpi_lib_handle;

/* ---- Load/unload ---- */

static int nvmpi_dynlink_load(void)
{
    if (nvmpi_lib_handle)
        return 0;

    nvmpi_lib_handle = dlopen("libnvmpi.so", RTLD_LAZY);
    if (!nvmpi_lib_handle)
        return -1;

#define LOAD_SYM(name)                                                  \
    do {                                                                \
        nvmpi_##name = (pf_nvmpi_##name)dlsym(nvmpi_lib_handle,        \
                                               "nvmpi_" #name);        \
        if (!nvmpi_##name) {                                            \
            dlclose(nvmpi_lib_handle);                                  \
            nvmpi_lib_handle = NULL;                                    \
            return -1;                                                  \
        }                                                               \
    } while (0)

    LOAD_SYM(create_decoder);
    LOAD_SYM(decoder_put_packet);
    LOAD_SYM(decoder_get_frame);
    LOAD_SYM(decoder_get_frame_fd);
    LOAD_SYM(decoder_close);
    LOAD_SYM(create_encoder);
    LOAD_SYM(encoder_put_frame);
    LOAD_SYM(encoder_put_frame_fd);
    LOAD_SYM(encoder_get_packet);
    LOAD_SYM(encoder_dqEmptyPacket);
    LOAD_SYM(encoder_qEmptyPacket);
    LOAD_SYM(encoder_close);

#undef LOAD_SYM

    return 0;
}

static void nvmpi_dynlink_unload(void)
{
    if (nvmpi_lib_handle) {
        dlclose(nvmpi_lib_handle);
        nvmpi_lib_handle = NULL;
    }
}

#endif /* DYNLINK_NVMPI_H */
