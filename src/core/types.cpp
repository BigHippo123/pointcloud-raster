#include "pcr/core/types.h"
#include <proj.h>
#include <algorithm>

namespace pcr {

// ---------------------------------------------------------------------------
// DataType utilities
// ---------------------------------------------------------------------------

size_t data_type_size(DataType dt) {
    switch (dt) {
        case DataType::Float32:  return 4;
        case DataType::Float64:  return 8;
        case DataType::Int32:    return 4;
        case DataType::UInt32:   return 4;
        case DataType::Int16:    return 2;
        case DataType::UInt16:   return 2;
        case DataType::UInt8:    return 1;
    }
    return 0;  // unreachable
}

// ---------------------------------------------------------------------------
// BBox
// ---------------------------------------------------------------------------

void BBox::expand(double x, double y) {
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
}

void BBox::expand(const BBox& other) {
    if (!other.valid()) return;
    expand(other.min_x, other.min_y);
    expand(other.max_x, other.max_y);
}

bool BBox::contains(double x, double y) const {
    return x >= min_x && x <= max_x && y >= min_y && y <= max_y;
}

// ---------------------------------------------------------------------------
// CRS
// ---------------------------------------------------------------------------

bool CRS::is_projected() const {
    // Simple heuristic: check WKT for PROJCS or PROJCRS
    if (wkt.find("PROJCS") != std::string::npos ||
        wkt.find("PROJCRS") != std::string::npos) {
        return true;
    }

    // Could also use PROJ library for more robust check
    if (!wkt.empty()) {
        PJ_CONTEXT* ctx = proj_context_create();
        PJ* pj = proj_create(ctx, wkt.c_str());
        if (pj) {
            PJ_TYPE type = proj_get_type(pj);
            proj_destroy(pj);
            proj_context_destroy(ctx);
            return type == PJ_TYPE_PROJECTED_CRS;
        }
        proj_context_destroy(ctx);
    }

    return false;
}

bool CRS::is_geographic() const {
    // Simple heuristic: check WKT for GEOGCS or GEOGCRS
    if (wkt.find("GEOGCS") != std::string::npos ||
        wkt.find("GEOGCRS") != std::string::npos) {
        return true;
    }

    // Could also use PROJ library for more robust check
    if (!wkt.empty()) {
        PJ_CONTEXT* ctx = proj_context_create();
        PJ* pj = proj_create(ctx, wkt.c_str());
        if (pj) {
            PJ_TYPE type = proj_get_type(pj);
            proj_destroy(pj);
            proj_context_destroy(ctx);
            return type == PJ_TYPE_GEOGRAPHIC_2D_CRS ||
                   type == PJ_TYPE_GEOGRAPHIC_3D_CRS;
        }
        proj_context_destroy(ctx);
    }

    return false;
}

CRS CRS::from_epsg(int code) {
    CRS crs;
    crs.epsg = code;

    // Use PROJ to get WKT from EPSG code
    PJ_CONTEXT* ctx = proj_context_create();
    std::string auth = "EPSG:" + std::to_string(code);
    PJ* pj = proj_create(ctx, auth.c_str());

    if (pj) {
        const char* wkt_cstr = proj_as_wkt(ctx, pj, PJ_WKT2_2019, nullptr);
        if (wkt_cstr) {
            crs.wkt = wkt_cstr;
        }
        proj_destroy(pj);
    }

    proj_context_destroy(ctx);
    return crs;
}

CRS CRS::from_wkt(const std::string& wkt_str) {
    CRS crs;
    crs.wkt = wkt_str;

    // Try to extract EPSG code from WKT if possible
    PJ_CONTEXT* ctx = proj_context_create();
    PJ* pj = proj_create(ctx, wkt_str.c_str());

    if (pj) {
        const char* auth_name = nullptr;
        const char* code_str = nullptr;
        if (proj_get_id_auth_name(pj, 0) && proj_get_id_code(pj, 0)) {
            auth_name = proj_get_id_auth_name(pj, 0);
            code_str = proj_get_id_code(pj, 0);

            if (auth_name && code_str && std::string(auth_name) == "EPSG") {
                crs.epsg = std::stoi(code_str);
            }
        }
        proj_destroy(pj);
    }

    proj_context_destroy(ctx);
    return crs;
}

bool CRS::equivalent_to(const CRS& other) const {
    // Quick check: same EPSG code
    if (epsg != 0 && other.epsg != 0 && epsg == other.epsg) {
        return true;
    }

    // Use PROJ to compare CRS definitions
    if (!wkt.empty() && !other.wkt.empty()) {
        PJ_CONTEXT* ctx = proj_context_create();
        PJ* pj1 = proj_create(ctx, wkt.c_str());
        PJ* pj2 = proj_create(ctx, other.wkt.c_str());

        bool equiv = false;
        if (pj1 && pj2) {
            equiv = proj_is_equivalent_to(pj1, pj2, PJ_COMP_EQUIVALENT) != 0;
        }

        if (pj1) proj_destroy(pj1);
        if (pj2) proj_destroy(pj2);
        proj_context_destroy(ctx);

        return equiv;
    }

    return false;
}

} // namespace pcr
