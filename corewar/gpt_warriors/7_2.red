
;name Skipper Bomber Optimized v5
;author ChatGPT
;strategy
;  Improved pacing by resetting counter explicitly.
;  Reduced overhead by using DJN with explicit reset and tighter loops.
;  Bombing uses post-increment indirect addressing to bomb then advance pointer.
;  Threads spawned once, then controlled internally.

        ORG start

step    EQU 4               ; step size to advance pointer
limit   EQU 18              ; countdown limit for pacing

target  DAT 0, 0            ; pointer to target instruction
counter DAT limit, 0        ; pacing counter

start   SPL bomb            ; spawn bombing thread once
        SPL advance         ; spawn advancing thread once
        JMP $+1             ; yield to spawned threads (do nothing here)

advance DJN counter, #-1    ; decrement counter and jump if not zero
        JMP advance         ; continue counting down

        ADD.AB #step, target ; when counter hits zero, add step to target
        MOV #limit, counter  ; reset counter
        JMP advance         ; loop advance

bomb    MOV.AB #0, }target  ; bomb instruction pointed by target, then increment pointer
        JMP bomb            ; continuous bombing loop

        END start
