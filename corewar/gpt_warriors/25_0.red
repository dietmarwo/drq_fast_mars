
;name          Spiral Bomber Faster & Less Predictable
;author        ChatGPT (improved)
;strategy      Creates two bombing threads for faster coverage,
;              uses postincrement indirect addressing for dynamic targeting,
;              and adds a small decrement step to add unpredictability.

        ORG     start

step    EQU     5               ; main step size for bombing increments
negstep EQU     -1              ; small decrement step to add unpredictability

target  DAT.F   #0, #0          ; pointer to target location

start   SPL     bomb1           ; start first bombing thread
        SPL     bomb2           ; start second bombing thread for faster bombing
        ADD.AB  #step, target   ; increment target pointer by main step
        SUB.AB  #1, target      ; slight adjustment to target pointer for unpredictability
        JMP     start           ; repeat main loop

bomb1   MOV.AB  #0, {target    ; bomb using A-number predecrement indirect addressing
        ADD.AB  #negstep, target ; adjust target backwards for next bomb in this thread
        JMP     bomb1          ; continue bombing

bomb2   MOV.AB  #0, >target    ; bomb using B-number postincrement indirect addressing
        ADD.AB  #step, target   ; move forward by step for next bomb in this thread
        JMP     bomb2          ; continue bombing

        END     start
