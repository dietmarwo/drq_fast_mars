
;name Skipper Bomber Optimized v9
;author ChatGPT
;strategy
;  Improved pacing by using DJN with JMZ to reduce jumps.
;  Simplified advance loop with DJN controlling pacing better.
;  Bomb thread uses post-increment indirect for efficient bombing.
;  Single initial SPL to spawn both threads, then self-loop to save instructions.

        ORG start

step    EQU 4               ; step size to advance pointer
limit   EQU 18              ; countdown limit for pacing

target  DAT 0, 0            ; pointer to target instruction
counter DAT limit, 0        ; pacing counter

start   SPL advance         ; spawn advance thread once
        JMP bomb            ; start bomb thread immediately

advance DJN counter, #0      ; decrement counter, if zero continue to advance
        JMZ advance          ; skip next if counter zero (pacing)
        ADD.AB #step, target ; move target pointer forward by step
        MOV #limit, counter  ; reset counter
        JMP advance          ; continue advance loop

bomb    MOV.AB #0, }target  ; bomb target and post-increment pointer
        SPL bomb            ; spawn another bomb thread
        JMP bomb            ; continue bombing loop

        END start
