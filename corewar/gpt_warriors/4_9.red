
;name     Spiral Bomber Improved
;author   Creative Corewar Assistant
;strategy Spirals through memory bombing every 3rd instruction with optimized bombing and faster loop

        ORG start

step    EQU 3                   ; Step size for spiral bombing

start   ADD.AB  #step, pointer   ; Increment bombing pointer by step
        MOV.F   bomb, @pointer   ; Bomb the instruction at pointer (both fields)
        ADD.A   #4, bomb         ; Increase bomb A-number by 4 for faster mutation and confusion
        JMP     start            ; Tight loop back to start for speed

pointer DAT.F   #0, #0           ; Current bombing target (offset)
bomb    DAT.F   #0, #-1          ; Bomb instruction set to DAT with B-field -1 (kills task)

        END start
