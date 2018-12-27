//
//  MPSMLModel.swift
//  CoreML-MPS
//
//  Created by 谢宜 on 2018/12/20.
//  Copyright © 2018 xieyi. All rights reserved.
//

import Foundation
import UIKit
import MetalPerformanceShaders
import MetalKit

public class MPSMLModel {
    
    let rawModel: RawMLModel
    
    var graph: MPSNNGraph!
    
    public init(model: String, bundle: Bundle) throws {
        rawModel = try RawMLModel(model, bundle)
    }
    
    public func createGraph(device: MTLDevice, input: String, output: String) throws {
        
        try rawModel.load()
        try rawModel.construct()
        
        // Construct graph
        let inputImage = MPSNNImageNode(handle: nil)
        
        var node: MLNode? = rawModel.graph[input]
        var image: MPSNNImageNode = inputImage
        var mpsDes: MPSDS? = nil
        var deconv: Bool = false
        
        guard node != nil else {
            throw MLModelError.nodeNotExist("Input feature \(input) does not exist")
        }
        
        // Prevent conv from deiniting
        var convs = [MPSNNFilterNode]()
        
        while node != nil && node?.name != output {
            let name = (node?.name)!
            let layer = rawModel.layers[name]
            if layer == nil {
                node = node?.top
                continue
            }
            let type = layer!["type"].stringValue
            if type == "activation" {
                if mpsDes != nil {
                    mpsDes?.addRelu(alpha: layer!["alpha"].floatValue)
                    var conv: MPSNNFilterNode
                    if deconv {
                        conv = MPSCNNConvolutionTransposeNode(source: image, weights: mpsDes!)
                    } else {
                        conv = MPSCNNConvolutionNode(source: image, weights: mpsDes!)
                    }
                    convs.append(conv)
                    image = conv.resultImage
                    mpsDes = nil
                } else {
                    throw MLModelError.unsupportedFormat("No conv before \(name)")
                }
            } else if type == "convolution" || type == "deconvolution" {
                if mpsDes != nil {
                    var conv: MPSNNFilterNode
                    if deconv {
                        conv = MPSCNNConvolutionTransposeNode(source: image, weights: mpsDes!)
                    } else {
                        conv = MPSCNNConvolutionNode(source: image, weights: mpsDes!)
                    }
                    convs.append(conv)
                    image = conv.resultImage
                }
                let w = rawModel.blobs[layer!["blob_weights"].intValue]
                var b: Data? = nil
                if layer!["has_biases"].intValue == 1 {
                    b = rawModel.blobs[layer!["blob_biases"].intValue]
                }
                let kw = layer!["Nx"].intValue
                let kh = layer!["Ny"].intValue
                let ifc = layer!["K"].intValue
                let ofc = layer!["C"].intValue
                deconv = (type == "deconvolution")
                mpsDes = MPSDS(label: name, w: w, b: b, kw: kw, kh: kh, ifc: ifc, ofc: ofc, deconv: deconv)
            } else {
                throw MLModelError.unsupportedFormat("Layer type \(type) unsupported")
            }
            node = node?.top
        }
        
        guard node?.name == output else {
            throw MLModelError.nodeNotExist("Out feature \(output) does not exist on path from input \(input)")
        }
        
        // Output layer
        if mpsDes != nil {
            var conv: MPSNNFilterNode
            if deconv {
                conv = MPSCNNConvolutionTransposeNode(source: image, weights: mpsDes!)
            } else {
                conv = MPSCNNConvolutionNode(source: image, weights: mpsDes!)
            }
            convs.append(conv)
            image = conv.resultImage
        }
        
        guard let graph = MPSNNGraph(device: device, resultImage: image, resultImageIsNeeded: true) else {
            throw MLModelError.cannotCreateGraph("Failed to create MPSNN graph for \(rawModel.name)")
        }
        self.graph = graph
    }
    
    public func encode(to commandBuffer: MTLCommandBuffer, input: MPSImage) -> MPSImage? {
        return self.graph.encode(to: commandBuffer, sourceImages: [input])
    }
    
}
