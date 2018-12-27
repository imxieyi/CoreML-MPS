//
//  fillalpha.metal
//  Demo
//
//  Created by 谢宜 on 2018/12/27.
//  Copyright © 2018 xieyi. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void FillAlphaMain(texture2d<float, access::read>  in  [[texture(0)]],
                         texture2d<float, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    float4 inc = in.read(gid);
    out.write(float4(inc[0], inc[1], inc[2], 1.0f), gid);
}
