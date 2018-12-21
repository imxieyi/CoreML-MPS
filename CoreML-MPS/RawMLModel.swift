//
//  RawMLModel.swift
//  CoreML-MPS
//
//  Created by 谢宜 on 2018/12/20.
//  Copyright © 2018 xieyi. All rights reserved.
//

import Foundation

class RawMLModel {
    
    let name: String
    private let netFile: URL?
    private let shapeFile: URL?
    private let weightsFile: URL?
    
    private(set) var shapes = [String : [String : Int]]()
    private(set) var layers = [String : JSON]()
    private(set) var blobs = [Data]()
    
    private(set) var graph = [String: MLNode]()
    
    init(_ name: String, _ bundle: Bundle) throws {
        self.name = name
        netFile = bundle.url(forResource: "model.espresso", withExtension: "net", subdirectory: name + ".mlmodelc")
        guard netFile != nil else {
            throw MLModelError.fileNotFound("model.espresso.net not found")
        }
        shapeFile = bundle.url(forResource: "model.espresso", withExtension: "shape", subdirectory: name + ".mlmodelc")
        guard shapeFile != nil else {
            throw MLModelError.fileNotFound("model.espresso.shape not found")
        }
        weightsFile = bundle.url(forResource: "model.espresso", withExtension: "weights", subdirectory: name + ".mlmodelc")
        guard weightsFile != nil else {
            throw MLModelError.fileNotFound("model.espresso.weights not found")
        }
    }
    
    func load() throws {
        // Load shapes
        let shapeData = FileManager.default.contents(atPath: (shapeFile?.path)!)
        guard shapeData != nil else {
            throw MLModelError.invalidFile("Failed to load model.espresso.shape")
        }
        let shapeObj = try JSON(data: shapeData!)["layer_shapes"]
        shapeObj.forEach { (k, v) in
            shapes[k] = [String : Int]()
            v.forEach({ (arg) in
                let (ik, iv) = arg
                shapes[k]![ik] = iv.intValue
            })
        }
        debugPrint("#shapes: \(shapes.count)")
        // Load net
        let netData = FileManager.default.contents(atPath: (netFile?.path)!)
        guard netData != nil else {
            throw MLModelError.invalidFile("Failed to load model.espresso.net")
        }
        let netObj = try JSON(data: netData!)
        if netObj["format_version"] != 200 {
            throw MLModelError.unsupportedFormat("Unsupported format: \(netObj["format_version"])")
        }
        netObj["layers"].array?.forEach({ (obj) in
            layers[obj["name"].stringValue] = obj
        })
        debugPrint("#layers: \(layers.count)")
        // Load blob data
        var weightsData = FileManager.default.contents(atPath: (weightsFile?.path)!)
        guard weightsData != nil else {
            throw MLModelError.invalidFile("Failed to load model.espresso.weights")
        }
        let blobCount: Int = weightsData?.withUnsafeBytes { $0.pointee } ?? 0
        guard blobCount > 0 else {
            throw MLModelError.invalidFile("Blob count is 0")
        }
        weightsData = weightsData?.advanced(by: 8)
        var blobBytes = [Int](repeating: 0, count: blobCount)
        for _ in 0..<blobCount {
            let idx: Int32 = weightsData?.withUnsafeBytes { $0.pointee } ?? 0
//            let idx = weightsData?.withUnsafeBytes { (p: UnsafePointer<Int32>) -> Int32 in
//                return p.pointee
//            }
            weightsData = weightsData?.advanced(by: 8)
            let length: Int32 = weightsData?.withUnsafeBytes { $0.pointee } ?? 0
            weightsData = weightsData?.advanced(by: 8)
            blobBytes[Int(idx)] = Int(length)
        }
        for i in 0..<blobCount {
            weightsData?.withUnsafeBytes { (pointer: UnsafePointer<UInt8>) -> Void in
                blobs.append(Data(buffer: UnsafeBufferPointer(start: pointer, count: blobBytes[i])))
            }
            if i < blobCount-1 {
                weightsData = weightsData?.advanced(by: blobBytes[i])
            }
        }
        debugPrint("#blobs: \(blobCount)")
    }
    
    func construct() throws {
        layers.forEach { (k, v) in
            let name = k
            let top = v["top"].stringValue
            let bottom = v["bottom"].stringValue
            var thisNode = graph[name]
            if thisNode == nil {
                thisNode = MLNode(name)
                graph[name] = thisNode
            }
            var topNode = graph[top]
            if topNode == nil {
                topNode = MLNode(top)
                graph[top] = topNode
            }
            var bottomNode = graph[bottom]
            if bottomNode == nil {
                bottomNode = MLNode(bottom)
                graph[bottom] = bottomNode
            }
            thisNode?.top = topNode
            topNode?.bottom = thisNode
            thisNode?.bottom = bottomNode
            bottomNode?.top = thisNode
        }
        graph.forEach { (k, v) in
            if v.bottom == nil {
                var node: MLNode? = v
                while node != nil {
                    debugPrint(node!.name, separator: "", terminator: "->")
                    node = node?.top
                }
                debugPrint("")
            }
        }
    }
    
}
