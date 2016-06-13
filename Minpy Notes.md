# Minpy Notes

## Core

- grad_and_loss
  - first, convert all the array to a `minpy type`
  - mark the array that needs BP 
  - then forward and get result array
  - use method of array to get gradient: array.node.partial_derivate(result)
  - ​