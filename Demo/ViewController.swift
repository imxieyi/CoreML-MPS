//
//  ViewController.swift
//  Demo
//
//  Created by 谢宜 on 2018/12/21.
//  Copyright © 2018 xieyi. All rights reserved.
//

import UIKit
import MetalPerformanceShaders
import MetalKit
import CoreML_MPS

class ViewController: UIViewController {

    @IBOutlet weak var timeLabel: UILabel!
    @IBOutlet weak var beforeView: UIImageView!
    @IBOutlet weak var afterView: UIImageView!
    @IBOutlet weak var runButton: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        beforeView.image = UIImage(named: "test.png")
    }
    
    func alert(_ message: String) {
        DispatchQueue.main.async {
            let alert = UIAlertController(title: "Error", message: message, preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "Close", style: .default, handler: { (_) in
                alert.dismiss(animated: true, completion: nil)
            }))
        }
    }
    
    @IBAction func runClicked(_ sender: Any) {
        runButton.isEnabled = false
        let cgimg = self.beforeView.image?.cgImage
        DispatchQueue(label: "model").async {
            let device = MTLCreateSystemDefaultDevice()
            guard MPSSupportsMTLDevice(device) else {
                self.alert("MPS not supported! Maybe you are using a simulator.")
                return
            }
            let commandQueue = device?.makeCommandQueue()
            // Create kernels
            let scaleKernel = MPSImageLanczosScale(device: device!)
            let scaleModel = try! MPSMLModel(model: "anime_scale2x_model", bundle: Bundle(for: ViewController.self))
            let noiseModel = try! MPSMLModel(model: "anime_noise3_model", bundle: Bundle(for: ViewController.self))
            let fillKernel = FillAlpha(device: device!)
            // Convert image
            let startTime = Date().timeIntervalSinceReferenceDate
            let textureLoader = MTKTextureLoader(device: device!)
            let texture = try! textureLoader.newTexture(cgImage: cgimg!, options: [
                // https://github.com/hollance/VGGNet-Metal/blob/master/VGGNet-iOS/VGGNet/ViewController.swift#L127
                MTKTextureLoader.Option.SRGB : NSNumber(value: false)
                ])
            let mpsimg = MPSImage(texture: texture, featureChannels: cgimg!.bitsPerPixel / cgimg!.bitsPerComponent)
            try! noiseModel.createGraph(device: device!, input: "input", output: "conv7")
            try! scaleModel.createGraph(device: device!, input: "input", output: "conv7")
            // Encode to kernels
            let outw = mpsimg.width * 2
            let outh = mpsimg.height * 2
            let outDesc = MPSImageDescriptor(channelFormat: .float16, width: outw, height: outh, featureChannels: 3)
            guard let commandBuffer = commandQueue?.makeCommandBuffer() else {
                self.alert("Failed to create command buffer.")
                return
            }
            let denoisedImage = noiseModel.encode(to: commandBuffer, input: mpsimg)
            let scaledImage = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: outDesc)
            scaleKernel.encode(commandBuffer: commandBuffer, sourceImage: denoisedImage!, destinationImage: scaledImage)
            let modelOutputImage = scaleModel.encode(to: commandBuffer, input: scaledImage)
            let outputImage = MPSImage(device: device!, imageDescriptor: outDesc)
            fillKernel.encode(commandBuffer: commandBuffer, sourceImage: modelOutputImage!, destinationImage: outputImage)
            // End encoding
            let endEncodingTime = Date().timeIntervalSinceReferenceDate
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            // Get output image
            let outTexture = outputImage.texture
            let outImage = UIImage.image(texture: outTexture)
            let endTime = Date().timeIntervalSinceReferenceDate
            DispatchQueue.main.async {
                let encodingTime = endEncodingTime - startTime
                let runTime = endTime - endEncodingTime
                self.timeLabel.text = "\(encodingTime),\(runTime)"
                self.runButton.isEnabled = true
                self.afterView.image = outImage
            }
        }
    }
    
}

