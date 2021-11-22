//
//  NeuralNetworkCore.swift
//  NeuralNetwork
//
//  Created by Sergey Reznichenko on 20.11.2021.
//

import Foundation
class NeuralNetworkCore {
    
    typealias DataSet = [DataSetPair]
    private let activationFunction: ActivationFunction
    private var layers: [NeuralLayer] = []
    private var learningRate: Double
    
    init(activationFunction: ActivationFunction, learningRate: Double) {
        self.activationFunction = activationFunction
        self.learningRate = learningRate
    }
    
    func addLayer(inputsCount: Int, neuronsCount: Int) {
        let layer = NeuralLayer(inputsCount: inputsCount, neuronsCount: neuronsCount, activFunc: activationFunction)
        layers.append(layer)
    }
    
    func feedForward(inputs: [Double]) -> [Double] {
        var inputValues = inputs
        for layer in layers {
            inputValues = layer.feedForward(inputs: inputValues)
        }
        return inputValues
    }
    
    func feedBackward(targets: [Double]) {
        
        for i in stride(from: 0, to: layers.last!.neuronsCount, by: 1) {
            let neuron: Neuron = layers.last!.neuronsImmutable[i]
            let error = -(targets[i] - neuron.output)
            neuron.calculateDelta(error: error)
        }
        // Hidden layers
        let hiddenLayersCount: Int = layers.count - 2
        var l = hiddenLayersCount - 1
        while l >= 0 {
            let currentLayer = self.layers[l]
            let lastLayer = self.layers[l + 1]
            for i in stride(from: 0, to: currentLayer.neuronsCount, by: 1) {
                let currentNeuron = currentLayer.neuronsImmutable[i]
                var totalError: Double = 0
                for j in stride(from: 0, to: lastLayer.neuronsCount, by: 1) {
                    let lastNeuron = lastLayer.neuronsImmutable[j]
                    totalError += lastNeuron.delta * lastNeuron.weights[i]
                }
                currentNeuron.calculateDelta(error: totalError)
            }
            l -= 1
        }
    }
    
    func updateWeights() {
        for layer in layers {
            layer.updateWeights(learningRate: learningRate)
        }
    }
    
    func calculateSingleError(targets: [Double], actualOutputs: [Double]) -> Double {
        var error: Double = 0
        for i in stride(from: 0, to: targets.count, by: 1) {
            error += pow((targets[i] - actualOutputs[i]), 2)
        }
        return error
    }
    
    func calculateTotalError(dataset: DataSet) -> Double {
        var totalError: Double = 0
        for value in dataset {
            let outputs: [Double] = self.feedForward(inputs: value.inputs)
            totalError += self.calculateSingleError(targets: value.targets, actualOutputs: outputs)
        }
        return totalError / Double(dataset.count)
    }
    
    func train(dataset: DataSet, iterationNumbers: Int) {
        print("-------Training-------")
        var totalError: Double = 0
        for i in stride(from: 0, to: iterationNumbers, by: 1) {
            print("data number:\(i)")
            for j in stride(from: 0, to: dataset.count, by: 1) {
                let pair = dataset[j]
                _ = feedForward(inputs: pair.inputs)
                feedBackward(targets: pair.targets)
                updateWeights()
            }
            totalError = calculateTotalError(dataset: dataset)
            print("Total error: \(totalError)")
        }
        print("-------Training Finished. Error: \(totalError)-------")
    }
    
    func test(dataset: DataSet) {
        print("-------Testing-------")
        for i in stride(from: 0, to: dataset.count, by: 1) {
            print("----data #\(i)----")
            let pair = dataset[i]
            let actualOutputs = self.feedForward(inputs: pair.inputs)
            for i in stride(from: 0, to: actualOutputs.count, by: 1) {
                print(" #\(i) \(actualOutputs[i]) is \(pair.targets[i])")
            }
        }
        let totalError = calculateTotalError(dataset: dataset)
        print("-------Testing Finished. Error: \(totalError)-------")
    }
    
    struct DataSetPair {
        var inputs: [Double]
        var targets: [Double]
    }
}

struct ActivationFunction {
    var fun: (Double) -> Double
    var dfun: (Double) -> Double
    
    static let sigmoid: ActivationFunction = ActivationFunction(fun: { val in
        return 1 / (1 + exp(-val))
    }, dfun: { val in
        val * (1 - val)
    })
    static let relu: ActivationFunction = ActivationFunction(fun: { val in
        if (val <= 0) {
            return val
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

class NeuralLayer {
    static var counter = 0
    private var neurons: [Neuron]
    init(inputsCount: Int, neuronsCount: Int, activFunc: ActivationFunction) {
        NeuralLayer.counter += 1
        self.neurons = []
        for i in stride(from: 0, to: neuronsCount, by: 1) {
            self.neurons.append(Neuron(weightsCount: neuronsCount, activFunc: activFunc))
        }
    }
    
    var neuronsImmutable: [Neuron] {
        return neurons
    }
    
    var neuronsCount: Int {
        neurons.count
    }
    
    
    func deltas() -> [Double] {
        return neurons.map { $0.delta }
    }
    
    func feedForward(inputs: [Double]) -> [Double] {
        return neurons.map { $0.calculateOutput(inputs: inputs) }
    }
    
    func updateWeights(learningRate: Double) {
        neurons.forEach {
            $0.updateWeights(learningRate: learningRate)
        }
    }
}

class Neuron {
    private(set) var bias: Double
    private(set) var output: Double = 0
    private(set) var delta: Double = 0.0
    private(set) var weightsCount: Int
    private let activFunc: ActivationFunction
    private var inputs: [Double] = []
    var weights: [Double]
    
    init(weightsCount: Int, activFunc: ActivationFunction, bias: Double = 1) {
        weights = Array(repeating: 0, count: weightsCount)
            .map { _ in Double.random(in: 0..<1) }
        self.weightsCount = weightsCount
        self.bias = bias
        self.activFunc = activFunc
    }
    
    func calculateOutput(inputs: [Double]) -> Double {
        output = zip(inputs, weights).reduce(into: 0, { $0 += $1.0 * $1.1 })
        let outputVal = activFunc.fun(output + bias)
        
        self.inputs = inputs
        self.output = outputVal
        return outputVal
    }
    
    func calculateDelta(error: Double) {
        delta = error * activFunc.dfun(self.output)
    }
    
    func updateWeights(learningRate: Double) {
        for i in stride(from: 0, to: min(weightsCount, self.inputs.count), by: 1) {
            self.weights[i] -= learningRate * self.delta * self.inputs[i]
        }
        self.bias -= learningRate * self.delta
        self.calculateOutput(inputs: inputs)
    }
}

extension Array where Element == Array<Array<Double>> {
    func toDataSet() -> NeuralNetworkCore.DataSet {
        map { NeuralNetworkCore.DataSetPair(inputs: $0[0], targets: $0[1]) }
    }
}
