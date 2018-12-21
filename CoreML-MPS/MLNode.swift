//
//  MLNode.swift
//  CoreML-MPS
//
//  Created by 谢宜 on 2018/12/20.
//  Copyright © 2018 xieyi. All rights reserved.
//

import Foundation

class MLNode {
    
    var top: MLNode? = nil
    var bottom: MLNode? = nil
    var name: String
    
    init(_ name: String) {
        self.name = name
    }
    
}
