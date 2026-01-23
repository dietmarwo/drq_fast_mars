
;name  SneakyBomber
;author ChatGPT
;strategy Improved SneakyBomber
;          Uses indirect bombing for unpredictability and faster bombing.
;          Moves target pointer to create a wandering bombing pattern.
;          Uses SPL to create a spare process for continuous bombing.

        ORG     start

step    EQU     5             ; steps to jump each iteration
count   DAT     #8, 0        ; bomb counter

start   SPL.B   bomb          ; spawn a bomber process
        DJN.A   dec_loop, count   ; decrement bomb counter and jump to dec_loop if not zero
        JMP.A   advance           ; else advance forward

dec_loop DJN.B #1, dec_loop      ; delay loop to slow bombing rate
        JMP.A   start             ; return to main loop

bomb    MOV.I  bomb_target, @target  ; bomb the target indirectly
        ADD.AB #step, target        ; move target forward by step
        JMP.A   bomb                ; repeat bombing

advance ADD.AB #step, target      ; move target pointer forward faster during count zero period
        JMP.A   start             ; loop forever

bomb_target DAT  #0, #0           ; bombing location
target      DAT  #0, #0           ; target pointer initialized to current position

        END     start
