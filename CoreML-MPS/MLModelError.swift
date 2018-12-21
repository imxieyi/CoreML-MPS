//
//  MLModelError.swift
//  CoreML-MPS
//
//  Created by 谢宜 on 2018/12/20.
//  Copyright © 2018 xieyi. All rights reserved.
//

import Foundation

public enum MLModelError: Error {
    case fileNotFound(String)
    case invalidFile(String)
    case unsupportedFormat(String)
    case nodeNotExist(String)
    case cannotCreateGraph(String)
}
