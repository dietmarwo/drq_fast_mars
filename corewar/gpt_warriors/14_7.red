
;name Spiral Bomber
;author ChatGPT
;strategy
;  This program spirals through memory bombing instructions in a growing pattern.
;  It starts by bombing the next instruction, then jumps further each time, spiraling through core.
;  Uses indirect addressing and self-modifying code to increase bombing distance steadily.

        ORG     start

step    EQU     3               ; step to increase bombing distance every loop

start   MOV.I   #0, target       ; initialize target pointer to 0

loop    ADD.AB  #step, target    ; increase target pointer by step
        MOV.AB  #0, @target      ; bomb the instruction at indirect target
        JMP.A   loop             ; keep looping infinitely

target  DAT.F   #0, #0           ; pointer to target instruction

        END     start
