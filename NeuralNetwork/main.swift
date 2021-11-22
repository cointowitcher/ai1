//
//  main.swift
//  NeuralNetwork
//
//  Created by Sergey Reznichenko on 20.11.2021.
//

import Foundation

func x1() {
    let test = NeuralNetworkTest()
    test.testForwardPropagation()
    test.testBackwardPropagation()

    let neural = NeuralNetwork2(inputNeuronsCount: 5*5, hiddenNeuronsCount: 31, outputNeuronsCount: 10, learningRate: 0.001, activationFunc: .relu)
    neural.initialize()
    neural.train(epochs: 200, printFrequency: 50, inputsAndOutputs: generateTrain5x5())
    
    print(neural.predict(inputs: getTestx()))
    print(neural.predict(inputs: getTestNum7()))
    print(neural.predict(inputs: getTestNum9()))
}

func x2() {
    let neural = NeuralNetwork2(inputNeuronsCount: 2, hiddenNeuronsCount: 2, outputNeuronsCount: 2, learningRate: 0.1, activationFunc: .relu)
    neural.initialize()
    print(neural.predict(inputs: [2, 2]))
    neural.train(epochs: 2, printFrequency: 40, inputsAndOutputs: [
        ([1, 0], [1, 0]),
        ([5, 0], [1, 0]),
        ([7, 0], [1, 0]),
        ([0, 3], [0, 1]),
        ([0, 2], [0, 1]),
        ([0, 5], [0, 1]),
        ([2, 0], [1, 0])
    ])
    print(neural.predict(inputs: [0, 15]))
    print(neural.predict(inputs: [2, 0]))
    print(neural.predict(inputs: [0, 9]))
    print(neural.predict(inputs: [2, 3]))
}

func x4() {
    let abTestData = getAbTestData()
    let abSolutions = getAbSolutions()
    let neural = NeuralNetwork2(inputNeuronsCount: abTestData[0].count, hiddenNeuronsCount: abTestData[0].count + 5, outputNeuronsCount: abSolutions[0].count, learningRate: 0.001, activationFunc: .relu)
    neural.initialize()
    neural.train(epochs: 50, printFrequency: 1, inputsAndOutputs: abTestComponize(abTestData, abSolutions).map { $0 })
    while true {
        let str = readLine()!
        let x = str.filter { !$0.isWhitespace || !$0.isNewline }.components(separatedBy: ",").filter { $0 == "1.0" || $0 == "0.0" }.map { Double($0)! }
        print(neural.forwardPropagation(input: x))
    }
    let g = [0.0,1.0,1.0,1.0,1.0,0.0,
             1.0,1.0,0.0,0.0,1.0,1.0,
             1.0,0.0,0.0,0.0,0.0,1.0,
             1.0,0.0,0.0,0.0,0.0,1.0,
             1.0,0.0,0.0,0.0,0.0,1.0,
             1.0,1.0,0.0,0.0,0.0,1.0,
             0.0,1.0,0.0,0.0,0.0,1.0,
             0.0,0.0,1.0,1.0,1.0,1.0]
//    print(neural.forwardPropagation(input: g))
}

x4()


//print([0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0].count)
