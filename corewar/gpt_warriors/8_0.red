
;name Echo Blaster Optimized v3
;author Assistant
;strategy Continuous bombing every two cells with double splitting at start to ramp processes.
;          Uses predecrement indirect addressing to reduce self-overwrite and improve bombing efficiency.
;          Steps pointer before bombing to ensure bombs land accurately and avoid self-hits.

        ORG start

start   spl bomb                ; spawn bomb process 1
        spl bomb                ; spawn bomb process 2
        mov #2, ptr             ; initialize pointer to 2 to start bombing ahead safely
        jmp step                ; jump into stepping loop

bomb    mov.i #0, {ptr          ; bomb the cell pointed by ptr, predecrement indirect addressing to avoid self-hit
        jmp bomb                ; bomb continuously

step    add #2, ptr             ; advance pointer by 2 to keep pace with bombing
        jmp step                ; loop forever

ptr     dat #0, #0             ; pointer storage

        END
