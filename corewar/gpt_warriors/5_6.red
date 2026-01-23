
;name Cycle Killer v2
;author ChatGPT
;strategy Improved bombing warrior that quickly cycles through memory by 3,
; uses SPL to create a second thread for faster bombing, replacing target with DAT
; to kill apart from incrementing target pointer by 3 each time.

        ORG start

step    EQU 3                   ; Step size to cycle through core in increments of 3

target  DAT #0, #0             ; Target pointer for bombing

start   SPL bomb               ; Spawn new thread to bomb concurrently
        ADD.AB #step, target   ; Increment target pointer by step size
        JMP.A start            ; Loop forever cycling

bomb    MOV.B #0, @target      ; Bomb target instruction with DAT 0,0
        JMP.A start            ; Return to main loop

        END start
