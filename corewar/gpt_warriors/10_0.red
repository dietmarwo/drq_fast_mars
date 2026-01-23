
;name Bouncer
;author ChatGPT
;strategy
; This warrior bounces a bomb back and forth between two points in core memory,
; gradually moving the bomb's position inward, attempting to hit and disable
; an opponent that tries to occupy those positions.

        ORG start

step    EQU 3                    ; step size for moving the bomb
left    DAT 0, 0                ; left boundary pointer
right   DAT 20, 0               ; right boundary pointer

start   MOV.I  #left,     ptr       ; initialize ptr with left address
        MOV.I  #right,    limit     ; initialize limit with right address
loop    MOV.I  #0,       @ptr       ; bomb the current target (DAT #0,0)
        ADD.A   #step,   ptr        ; move ptr right by step
        CMP     ptr, limit          ; check if ptr reached or passed limit
        JMZ     swap, ptr           ; if equal, swap directions
        SPL     loop                ; continue looping
        JMP     loop

swap    MOV.I   limit, ptr         ; swap ptr and limit
        MOV.I   #left, limit       ; reset limit to left
        JMP     loop

ptr     DAT 0, 0
limit   DAT 0, 0

        END start
