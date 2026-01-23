
;name Improved Pulse Bomber v5
;author ChatGPT
;strategy Optimized multiple bomber with minimal SPL usage,
; uses DJN loop only to spawn extra tasks once, then bombs in parallel,
; predecrement indirect addressing for bombing ensures continual pointer stepping,
; less chance of task queue overflow, better survivability and efficiency

        ORG start

step    EQU 5               ; step size for bombing
count   EQU 3               ; number of bombing tasks
ptr     DAT #0, #0          ; pointer to target location

start   ADD #step, ptr       ; advance pointer by step
        MOV.I #0, {ptr       ; bomb current target with predecrement indirect addressing
        SPL bomb1            ; spawn first additional bombing task (only once)
        DJN count, spawn     ; spawn remaining tasks
        JMP start            ; continue main bombing loop

spawn   SPL bomb2            ; spawn second bombing task
        SPL bomb3            ; spawn third bombing task
        DAT #0, #0           ; end spawning task

bomb1   MOV.I #0, {ptr       ; bomb location pointed by ptr
        JMP bomb1            ; continuously bomb, no termination

bomb2   MOV.I #0, {ptr       ; bomb location pointed by ptr
        JMP bomb2            ; continuously bomb, no termination

bomb3   MOV.I #0, {ptr       ; bomb location pointed by ptr
        JMP bomb3            ; continuously bomb, no termination

        END start
