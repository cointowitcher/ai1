//
//  Utilities2.swift
//  NeuralNetwork
//
//  Created by Sergey Reznichenko on 22.11.2021.
//

import Foundation

func abTestComponize(_ data: [[Double]], _ sols: [[Double]]) -> [([Double], [Double])] {
    var output = zip(data, sols)
    return output.reversed().reversed()
}

func testComponize2(_ data: [[Double]], _ sols: [[Double]]) -> [[[Double]]] {
    var output = zip(data, sols).map { [$0.0, $0.1] }
    return output.reversed().reversed()
}
