
;name Spiral Bomber v4
;author ChatGPT improved
;strategy
; Improved Spiral Bomber with faster target stepping and efficient bombing.
; Removes JMP start infinite spawn loop for controlled initial tasks.
; Uses DJN to pace bombing and avoid task explosion.
; Two spirals start at different points for better coverage stepping by 3.

        ORG start

step    EQU 3                 ; step size for spiral bombing

target  DAT #0, #0            ; pointer to current bomb target

start   MOV target, bombers   ; initialize bombers' pointers
        SPL spiral1           ; spawn first bombing spiral
        SPL spiral2           ; spawn second bombing spiral
        JMP wait              ; wait loop (idling main task)

bombers DAT #0, #0            ; shared target pointer storage

wait    JMP wait              ; idle main task

spiral1 MOV.AB #0, >bombers   ; bomb current target and post-increment pointer
        ADD.AB #step, bombers ; advance pointer by step
        DJN.A  #10, spiral1   ; bomb 10 times then terminate this spiral

spiral2 ADD.AB #step, bombers ; start offset target pointer by step
        MOV.AB #0, >bombers   ; bomb current target and post-increment pointer
        ADD.AB #step, bombers ; advance pointer by step
        DJN.A  #10, spiral2   ; bomb 10 times then terminate spiral

        END
