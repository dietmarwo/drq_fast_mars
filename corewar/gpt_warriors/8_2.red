
;name Echo Blaster Optimized v7
;author Assistant
;strategy Continuous bombing every two cells with triple splitting at start for more processes.
;          Uses predecrement indirect addressing to reduce self-overwrite and improve bombing efficiency.
;          Steps pointer before bombing to ensure bombs land accurately and avoid self-hits.
;          Adds NOP in bombing loop to avoid rapid self-conflicts and improve process cycling.
;          Reordered bombing and stepping for better synchronization and survival chances.
;          Changed step to jmp 0 to keep main process active and avoid deadlock.

        ORG start

start   spl bomb                ; spawn bomb process 1
        spl bomb                ; spawn bomb process 2
        spl bomb                ; spawn bomb process 3 for more parallel attacks
        mov #2, ptr             ; initialize pointer to 2 to start bombing ahead safely
        jmp step                ; jump to stepping loop

bomb    add #-1, ptr            ; decrement pointer first (predecrement)
        mov.i #0, {ptr          ; bomb the cell at ptr (now decremented)
        nop                     ; slight delay to reduce overwrite conflicts
        add #2, ptr             ; advance pointer inside bomb process for synchronized bombing
        jmp bomb                ; continue bombing loop

step    jmp 0                   ; keep main process active by jumping to itself, avoiding dead task queue

ptr     dat #0, #0             ; pointer storage

        END
