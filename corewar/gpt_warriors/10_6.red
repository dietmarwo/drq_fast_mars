
;name Quick Bomber Optimized v4
;author ChatGPT
;strategy Dual-process quick bomber with improved survival and bombing efficiency.
;          Uses SPL once, main process increments pointer and skips bombing when target is zero,
;          bombing uses post-increment indirect addressing to avoid repeated decrements.
;          Uses JMZ for clearer zero check, reducing unnecessary bombing and improving performance.

        ORG start

step    EQU 4                  ; bomb every 4 instructions

start   SPL bomb               ; spawn parallel bombing process
        ADD.AB #step, pointer  ; increment pointer by step in main process
        JMZ.B  start, @pointer  ; if target is zero (B-field indirect), skip bombing and loop
        MOV.AB #0, }pointer    ; bomb target with post-increment indirect addressing
        JMP start              ; loop main process

bomb    ADD.AB #step, pointer  ; increment pointer in bombing process
        MOV.AB #0, }pointer    ; bomb target with post-increment indirect addressing
        JMP bomb               ; loop bombing process

pointer DAT #0, #0            ; bombing pointer initialized

        END start
