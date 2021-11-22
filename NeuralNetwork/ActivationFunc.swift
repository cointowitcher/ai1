//
//  ActivationFunc.swift
//  NeuralNetwork
//
//  Created by Sergey Reznichenko on 21.11.2021.
//

import Foundation

struct ActivationFunc {
    var fun: (Double) -> Double
    var dfun: (Double) -> Double
    
    static let sigmoid: ActivationFunc = ActivationFunc(fun: { val in
        return 1 / (1 + exp(-val))
    }, dfun: { val in
        val * (1 - val)
    })
    
    static let relu: ActivationFunc = ActivationFunc(fun: { val in
        if (val <= 0) {
            return 0
        } else {
            return val
        }
    }, dfun: { val in
        if (val <= 0) {
            return 0
        } else {
            return 1
        }
    })
}
