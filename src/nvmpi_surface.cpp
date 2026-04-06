/*
 * nvmpi_surface.cpp — C wrappers for NvBufSurface allocation
 *
 * Provides encoder-compatible buffer allocation via NvBufSurf::NvAllocate
 * with proper memtag. Used by the VIC scale filter (vf_scale_vic.c).
 */

#include "nvmpi.h"
#include "nvUtils2NvBuf.h"
#include <cstring>

extern "C" {

int nvmpi_surface_alloc(unsigned int width, unsigned int height,
    int color_format, int layout, int mem_type,
    int *dmabuf_fd, void **surf_out)
{
    NvBufSurf::NvCommonAllocateParams params;
    memset(&params, 0, sizeof(params));
    params.width       = width;
    params.height      = height;
    params.colorFormat = (NvBufSurfaceColorFormat)color_format;
    params.layout      = (NvBufSurfaceLayout)layout;
    params.memType     = (NvBufSurfaceMemType)mem_type;
    params.memtag      = NvBufSurfaceTag_VIDEO_ENC;

    int fd = -1;
    int ret = NvBufSurf::NvAllocate(&params, 1, &fd);
    if (ret < 0)
        return ret;

    *dmabuf_fd = fd;

    if (surf_out) {
        NvBufSurface *surf = NULL;
        ret = NvBufSurfaceFromFd(fd, (void **)&surf);
        if (ret < 0) {
            NvBufSurf::NvDestroy(fd);
            *dmabuf_fd = -1;
            return ret;
        }
        *surf_out = surf;
    }

    return 0;
}

int nvmpi_surface_destroy(int dmabuf_fd)
{
    return NvBufSurf::NvDestroy(dmabuf_fd);
}

} /* extern "C" */
