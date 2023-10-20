# tic-tac-toe_CUDA
Developing a multi-GPU Tic-Tac-Toe game using CUDA for parallel computing and enhanced gaming experiences

## Prerequisites
```
  Ubuntu 22.04
  CUDA 12.3
```

## Project Structure
![Untitled drawing](https://github.com/Kaiwei0323/tic-tac-toe_CUDA/assets/91507316/411b9e40-0fec-4ea0-b310-f1a0700002d7)

## Algorthm
* The player will first choose the middle of the grid
* The player will choose a corner of the grid if the middle is occupied
* If another player is about to win, or if the player is about to win, the player will prioritize defence first or play for the win


## Result
```bash
Number of GPU Devices: 1
Device Number: 0
  Device name: NVIDIA GeForce RTX 3080 Laptop GPU
  Device Compute Major: 8 Minor: 6
  Max Thread Dimensions: [1024][1024][64]
  Max Threads Per Block: 1024
  Number of Multiprocessors: 48
  Device Clock Rate (KHz): 1365000
  Memory Bus Width (bits): 256
  Registers Per Block: 65536
  Registers Per Multiprocessor: 65536
  Shared Memory Per Block: 49152
  Shared Memory Per Multiprocessor: 102400
  Total Constant Memory (bytes): 65536
  Total Global Memory (bytes): 17179344896
  Warp Size: 32
  Peak Memory Bandwidth (GB/s): 384.064000

The chosen GPU device has an index of: 0
Result: Tie
X|X|O
-----
O|O|X
-----
X|O|O

Pres Any Key to Rematch Or Enter -1 to Exit
-------------------------------------------

```

