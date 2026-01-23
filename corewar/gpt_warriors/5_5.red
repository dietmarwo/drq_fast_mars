
;name Cycler Improved v2
;author Assistant
;strategy Spawns two processes per cycle for faster replication. Uses postincrement indirect addressing for rapid copying and safer replication.

        ORG start

start   SPL     #1             ; Spawn a new process at next instruction (immediate mode)
        SPL     #2             ; Spawn another process two instructions ahead
        MOV.I   }0, 1          ; Use A-number postincrement indirect on current line to copy instruction and increment pointer for rapid replication
        JMP     start          ; Loop forever

        END start
