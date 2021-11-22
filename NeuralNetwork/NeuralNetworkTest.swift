//
//  NeuralNetworkTest.swift
//  NeuralNetwork
//
//  Created by Sergey Reznichenko on 21.11.2021.
//

class NeuralNetworkTest {

    func testForwardPropagation() {
        let network = NeuralNetwork2(inputNeuronsCount: 2, hiddenNeuronsCount: 3, outputNeuronsCount: 2, activationFunc: .relu)
        network.setValuesForTest(hiddenWeights: [
            [1, 0.2, 0.4],
            [-1, 2, 0.5]
        ], outputWeights: [
            [2, 1],
            [5, 1],
            [0.1, -2]
        ])
        let output = network.forwardPropagation(input: [3, 0.4]).1
        let expected = [12.34, 1.2]
        if (abs(output[0] - expected[0]) < 0.01 && abs(output[1] - expected[1]) < 0.01) {
            print("Success")
        } else {
            print("Error: \(output) expected to be: \(expected)")
        }
    }
    
    func testBackwardPropagation() {
        let network = NeuralNetwork2(inputNeuronsCount: 2, hiddenNeuronsCount: 3, outputNeuronsCount: 2, activationFunc: .relu)
        network.setValuesForTest(hiddenWeights: [
            [1, 0.2, 0.4],
            [-1, 2, 0.5]
        ], outputWeights: [
            [2, 1],
            [5, 1],
            [0.1, -2]
        ])
        let inputs = [3, 0.4]
        let output = network.forwardPropagation(input: inputs)
        
        network.backwardPropagationWithCorrection(inputs: inputs, hiddenResults: output.0, outputResults: output.1, expectedResults: [5, 3])
//        if (abs(output[0] - expected[0]) < 0.01 && abs(output[1] - expected[1]) < 0.01) {
//            print("Success")
//        } else {
//            print("Error: \(output) expected to be: \(expected)")
//        }

    }

}
