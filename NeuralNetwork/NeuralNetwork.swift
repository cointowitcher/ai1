//
//  NeuralNetwork.swift
//  NeuralNetwork
//
//  Created by Sergey Reznichenko on 21.11.2021.
//

import Foundation

class NeuralNetwork2 {
    
    private let inputNeuronsCount: Int
    private let hiddenNeuronsCount: Int
    private let outputNeuronsCount: Int
    private let learningRate: Double
    private let activationFunc: ActivationFunc
    private var hiddenWeights: [[Double]]
    private var hiddenBiases: [Double]
    private var outputWeights: [[Double]]
    private var outputBiases: [Double]
    
    init(inputNeuronsCount: Int, hiddenNeuronsCount: Int, outputNeuronsCount: Int, learningRate: Double = 0.1, activationFunc: ActivationFunc) {
        self.inputNeuronsCount = inputNeuronsCount
        self.hiddenNeuronsCount = hiddenNeuronsCount
        self.outputNeuronsCount = outputNeuronsCount
        self.learningRate = learningRate
        self.activationFunc = activationFunc
        hiddenWeights = []
        outputWeights = []
        hiddenBiases = []
        outputBiases = []
    }
    
    func initialize() {
        hiddenWeights.reserveCapacity(inputNeuronsCount)
        for _ in range(stop: inputNeuronsCount) {
            var newArray = Array<Double>()
            newArray.reserveCapacity(hiddenNeuronsCount)
            for _ in range(stop: hiddenNeuronsCount) {
                newArray.append(Double.random(in: weightsRandomRange))
            }
            hiddenWeights.append(newArray)
        }
        outputWeights.reserveCapacity(hiddenNeuronsCount)
        for _ in range(stop: hiddenNeuronsCount) {
            var newArray = Array<Double>()
            newArray.reserveCapacity(outputNeuronsCount)
            for _ in range(stop: outputNeuronsCount) {
                newArray.append(Double.random(in: weightsRandomRange))
            }
            outputWeights.append(newArray)
        }
        hiddenBiases = Array(repeating: 1, count: hiddenNeuronsCount)
        outputBiases = Array(repeating: 1, count: outputNeuronsCount)
    }
    
    func forwardPropagation(input: [Double]) -> ([Double], [Double]) {
        // To hidden
        var toHiddenValues = [Double]()
        toHiddenValues.reserveCapacity(hiddenWeights.count)
        for i in range(stop: hiddenNeuronsCount) {
            var value: Double = 0
            for j in range(stop: inputNeuronsCount) {
                value += hiddenWeights[j][i] * input[j]
            }
            toHiddenValues.append(activationFunc.fun(value))
        }
        // To output
        var outputValues = [Double]()
        outputValues.reserveCapacity(outputWeights.count)
        for i in range(stop: outputNeuronsCount) {
            var value: Double = 0
            for j in range(stop: hiddenNeuronsCount) {
                value += outputWeights[j][i] * toHiddenValues[j]
            }
            outputValues.append(activationFunc.fun(value))
        }
        return (toHiddenValues, outputValues)
    }
    
    func backwardPropagationWithCorrection(inputs: [Double], hiddenResults: [Double], outputResults: [Double], expectedResults: [Double]) {
        var outputDeltas = [[Double]]()
        var weightsDeltas = [Double]()
        weightsDeltas.reserveCapacity(hiddenNeuronsCount)
        outputDeltas.reserveCapacity(hiddenNeuronsCount)
        for i in range(stop: hiddenNeuronsCount) {
            var deltas = [Double]()
            deltas.reserveCapacity(outputNeuronsCount)
            var deltaSum: Double = 0
            for j in range(stop: outputNeuronsCount) {
                let error = (expectedResults[j] - outputResults[j])
                let value: Double = outputWeights[i][j] * activationFunc.dfun(hiddenResults[i]) * error * learningRate
                deltas.append(value)
                deltaSum += value
            }
            weightsDeltas.append(deltaSum)
            outputDeltas.append(deltas)
        }
        
        var hiddenDeltas = [[Double]]()
        hiddenDeltas.reserveCapacity(inputNeuronsCount)
        for i in range(stop: inputNeuronsCount) {
            var deltas = [Double]()
            deltas.reserveCapacity(hiddenNeuronsCount)
            for j in range(stop: hiddenNeuronsCount) {
                deltas.append(hiddenWeights[i][j] * activationFunc.dfun(inputs[i]) * weightsDeltas[j] * learningRate)
            }
            hiddenDeltas.append(deltas)
        }
        
        // apply
        for i in range(stop: hiddenNeuronsCount) {
            for j in range(stop: outputNeuronsCount) {
                outputWeights[i][j] += outputDeltas[i][j]
            }
        }
        for i in range(stop: inputNeuronsCount) {
            for j in range(stop: hiddenNeuronsCount) {
                hiddenWeights[i][j] += hiddenDeltas[i][j]
            }
        }
    }
    
    func calculateSingleError(targets: [Double], actualOutputs: [Double]) -> Double {
        var error: Double = 0
        for i in stride(from: 0, to: targets.count, by: 1) {
            error += pow((targets[i] - actualOutputs[i]), 2)
        }
        return error
    }
    
    func train(epochs: Int = 20, printFrequency: Int = 1,  inputsAndOutputs: [([Double], [Double])]) {
        var x = 0
        for i in range(stop: epochs) {
            if x % printFrequency == 0 {
                
                print("---------------Epoch #\(i)------------------")
            }
            var j = -1
            x += 1
            for val in inputsAndOutputs {
                j += 1
                let input = val.0
                let expectedOutput = val.1
                let res1 = forwardPropagation(input: input)
                if x % printFrequency == 0 {
                    print("Error #\(j): \(calculateSingleError(targets: expectedOutput, actualOutputs: res1.1))")
                }
                backwardPropagationWithCorrection(inputs: input, hiddenResults: res1.0, outputResults: res1.1, expectedResults: expectedOutput)
            }
        }
    }
    
    func predict(inputs: [Double]) -> [Double] {
        return forwardPropagation(input: inputs).1
    }
    
    func setValuesForTest(hiddenWeights: [[Double]], outputWeights: [[Double]]) {
        self.hiddenWeights = hiddenWeights
        self.outputWeights = outputWeights
    }
}

fileprivate let weightsRandomRange: Range<Double> = 0..<1
