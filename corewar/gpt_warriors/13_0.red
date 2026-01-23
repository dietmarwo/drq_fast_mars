
;name    Simple Bomber
;author  ChatGPT
;strategy Bomb every 5th instruction to cause disruption

        ORG     start

step    EQU     5                ; Step size to move through memory
target  DAT.F   #0, #0           ; Holds the target address to bomb

start   ADD.AB  #step, target    ; Increment the target by step value
        MOV.AB  #0, @target      ; Bomb the instruction at target address (replace with DAT)
        JMP.A   start            ; Loop back to start

        END     start
