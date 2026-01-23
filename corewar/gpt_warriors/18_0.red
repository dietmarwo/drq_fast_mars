
;name Spiral Bomber Optimized v3
;author ChatGPT
;strategy Bombs in a spiral pattern using single task with split to maintain concurrency.
; Combines step increment and bombing efficiently, removes redundant fields.

        ORG start

step    DAT #1              ; step size (A-field)
target  DAT #0              ; pointer to target

start   SPL bomb            ; create second task to bomb target concurrently
        MOV step, tmp      ; copy step to tmp
        MOV target, tmp+1  ; copy target to tmp+1
        JMP control

bomb    ADD.AB tmp, tmp+1   ; increment target pointer by step in tmp+1 (B)
        MOV.I  #0, @tmp+1  ; bomb target
        ADD.A #1, tmp      ; increment step
        JMP bomb

control DJN count, start    ; loop decrementing count, synced with bombing task
        JMP done

count   DAT #10             ; bomb count

tmp     DAT 0, 0           ; temporary storage for step and target

done    JMP done            ; infinite loop to end program

        END start
