//
//  MPSDS.swift
//  CoreML-MPS
//
//  Created by 谢宜 on 2018/12/20.
//  Copyright © 2018 xieyi. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class MPSDS: NSObject, MPSCNNConvolutionDataSource {
    
    func copy(with zone: NSZone? = nil) -> Any {
        let copied = MPSDS(label: l, w: w_d, b: b_d, kw: kernelHeight, kh: kernelHeight, ifc: inputFeatureChannels, ofc: outputFeatureChannels)
        if w_data != nil && !copied.load() {
            fatalError("Failed to copy MPSDS!")
        }
        return copied
    }
    
    
    private(set) var w_data: UnsafeMutableRawPointer!
    private(set) var b_data: UnsafeMutablePointer<Float>?
    
    private(set) var w_d: Data!
    private(set) var b_d: Data?
    
    let kernelWidth: Int
    let kernelHeight: Int
    let inputFeatureChannels: Int
    let outputFeatureChannels: Int
    
    private(set) var xStride: Int
    private(set) var yStride: Int
    
    private(set) var relu: Bool
    private(set) var alpha: Float
    
    let l: String
    
    init(label: String, w: Data, b: Data?, kw: Int, kh: Int, ifc: Int, ofc: Int, deconv: Bool = false) {
        kernelWidth = kw
        kernelHeight = kh
        inputFeatureChannels = ifc
        outputFeatureChannels = ofc
        self.l = label
        var wT = Data(count: w.count)
        if deconv {
            // CoreML Format: [ifc ofc w h]
            // MPSNN  Format: [ofc h w ifc]
            w.withUnsafeBytes { (readP: UnsafePointer<Float32>) -> Void in
                wT.withUnsafeMutableBytes({ (writeP: UnsafeMutablePointer<Float32>) -> Void in
                    var idx = 0
                    for o in 0..<ofc {
                        for h in 0..<kh {
                            for w in 0..<kw {
                                for i in 0..<ifc {
                                    writeP[idx] = readP[h + w * kh + o * kw * kh + i * ofc * kw * kh]
                                    idx += 1
                                }
                            }
                        }
                    }
                })
            }
        } else {
            // CoreML Format: [ofc ifc w h]
            // MPSNN  Format: [ofc h w ifc]
            w.withUnsafeBytes { (readP: UnsafePointer<Float32>) -> Void in
                wT.withUnsafeMutableBytes({ (writeP: UnsafeMutablePointer<Float32>) -> Void in
                    var idx = 0
                    for o in 0..<ofc {
                        for h in 0..<kh {
                            for w in 0..<kw {
                                for i in 0..<ifc {
                                    writeP[idx] = readP[h + w * kh + i * kw * kh + o * ifc * kw * kh]
                                    idx += 1
                                }
                            }
                        }
                    }
                })
            }
        }
        w_d = wT
        b_d = b
        self.relu = false
        self.alpha = 0
        self.xStride = 1
        self.yStride = 1
        super.init()
    }
    
    func addRelu(alpha: Float) {
        self.relu = true
        self.alpha = alpha
    }
    
    func setStride(x: Int, y: Int) {
        self.xStride = x
        self.yStride = y
    }
    
    func dataType() -> MPSDataType {
        return .float32
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth, kernelHeight: kernelHeight, inputFeatureChannels: inputFeatureChannels, outputFeatureChannels: outputFeatureChannels)
        desc.strideInPixelsX = xStride
        desc.strideInPixelsY = yStride
        if relu {
            desc.setNeuronType(.reLU, parameterA: alpha, parameterB: 0)
        } else {
            desc.setNeuronType(.none, parameterA: 0, parameterB: 0)
        }
        return desc
    }
    
    func weights() -> UnsafeMutableRawPointer {
        return w_data
    }
    
    func biasTerms() -> UnsafeMutablePointer<Float>? {
        return b_data
    }
    
    func load() -> Bool {
        guard w_d.count == kernelWidth * kernelHeight * inputFeatureChannels * outputFeatureChannels * 4 else {
            return false
        }
        w_data = UnsafeMutableRawPointer(mutating: (w_d as NSData).bytes)
        if b_d != nil {
            guard b_d!.count == outputFeatureChannels * 4 else {
                return false
            }
            b_data = UnsafeMutablePointer<Float>(OpaquePointer(UnsafeMutableRawPointer(mutating: (b_d! as NSData).bytes)))
        }
        guard w_data != nil else {
            return false
        }
        return true
    }
    
    func purge() {
        w_data = nil
        b_data = nil
    }
    
    func label() -> String? {
        return l
    }
    
}
