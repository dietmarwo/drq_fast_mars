
;name  SneakyBomber
;author ChatGPT
;strategy Improved SneakyBomber
;          Uses indirect bombing for unpredictability and faster bombing.
;          Moves target pointer to create a wandering bombing pattern.
;          Uses SPL to create a spare process for continuous bombing.
;          Optimized delays and bombing loop

        ORG     start

step    EQU     4             ; smaller step for more frequent bombing
count   DAT     #16, 0       ; increased bomb counter for longer attack

start   SPL.B   bomb          ; spawn a bomber process
        DJN.A   count_loop, count   ; decrement bomb counter and jump to count_loop if not zero
        JMP.A   advance             ; else advance forward

count_loop DJN.B #2, count_loop  ; short delay loop to pace bombing
          JMP.A   start            ; return to main loop

bomb    MOV.I  bomb_target, @target  ; bomb the target indirectly
        ADD.AB #step, target        ; move target forward by step
        SPL      bomb              ; create spare process for more attacks
        JMP.A   bomb                ; continue bombing loop

advance ADD.AB #step, target      ; move target pointer forward faster during count zero period
        JMP.A   start             ; loop forever

bomb_target DAT  #0, #0           ; bombing location
target      DAT  #0, #0           ; target pointer initialized to current position

        END     start
