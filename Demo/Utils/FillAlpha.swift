//
//  FillAlpha.swift
//  Demo
//
//  Created by 谢宜 on 2018/12/27.
//  Copyright © 2018 xieyi. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class FillAlpha: MPSUnaryImageKernel {
    
    override func encode(commandBuffer: MTLCommandBuffer, sourceTexture: MTLTexture, destinationTexture: MTLTexture) {
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()
        let library = try! device.makeDefaultLibrary(bundle: Bundle(for: type(of: self)))
        let constants = MTLFunctionConstantValues()
        let sampleMain = try! library.makeFunction(name: "FillAlphaMain", constantValues: constants)
        let pipelineState = try! device.makeComputePipelineState(function: sampleMain)
        commandEncoder?.setComputePipelineState(pipelineState)
        commandEncoder?.setTexture(sourceTexture, index: 0)
        commandEncoder?.setTexture(destinationTexture, index: 1)
        let threadGroupCount = MTLSize(width: 1, height: 1, depth: 1)
        let threadGroups = MTLSize(width: destinationTexture.width / threadGroupCount.width, height: destinationTexture.height / threadGroupCount.height, depth: threadGroupCount.depth)
        commandEncoder?.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)
        commandEncoder?.endEncoding()
    }
    
}
