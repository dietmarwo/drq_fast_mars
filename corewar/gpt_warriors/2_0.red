
;name Cycle Bomber
;author ChatGPT
;strategy Bombs every third instruction cycling through memory

        ORG start

step    EQU 3                ; step size for bombing
target  DAT  #0, #0          ; pointer to the current bombing target

start   ADD.AB  #step, target  ; increment target by step (3)
        MOV.AB  #0, @target    ; overwrite the target instruction with DAT #0, #0 (bomb)
        JMP     start          ; loop infinitely

        END start
