
;name SpiralBomber Improved
;author ChatGPT
;strategy
; Increases bombing offset faster to cover more memory quickly.
; Uses post-increment indirect addressing for dynamic targeting.
; Adds SPL to create another thread, increasing speed of bombing.
; Uses DJN to limit bomb count, avoiding infinite self-damage.

        ORG start

step    EQU 7                   ; larger step size to increase bombing coverage
bomb_limit DAT 30, 0            ; limit number of bombs dropped

start   SPL fork                ; split to create a bombing thread
        DJN bomb_limit, start   ; decrement bomb count and loop until zero
        JMP endprog             ; terminate after bomb_limit drops

fork    ADD #step, offset       ; increase bombing offset by step size
        MOV bomb, @offset       ; bomb at computed address
        SPL fork2               ; spawn a second thread for faster bombing
        JMP fork                ; continue bombing in this thread

fork2   ADD #step, offset       ; second thread also increases bombing offset
        MOV bomb, @offset       ; bomb at computed address
        JMP fork2               ; loop in second thread

offset  DAT 0, 0                ; current bombing offset
bomb    DAT 0, 0                ; bomb instruction (kills process)
endprog DAT 0, 0                ; termination point (does nothing)

        END start
