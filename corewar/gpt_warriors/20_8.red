
;name MirrorBot Improved Optimized v4
;author CoreWarAssistant
;strategy Interleaved replicator and bomber with refined pointer wrapping using postincrement
;         Improved bombing increment to prime step for better dispersion
;         Use SPL.B to queue splitter tasks more efficiently and ensure task future

        ORG start

start   spl.b   copy          ; Start replicator process with B modifier for faster task queue
        spl.b   bomb          ; Start bomber process in parallel with B modifier
        jmp     loop          ; Main monitoring loop

copy    mov.i   0, }ptr       ; Replicate current instruction to address pointed by ptr with postincrement indirect
        jmn     copy, ptr     ; Loop until pointer wraps to zero (non-zero jump), replication complete

loop    jmz     start, ptr    ; If replication pointer wrapped to zero, restart processes (respawn)
        jmp     loop          ; Otherwise continue monitoring

bomb    mov.i   bomb, }bptr   ; Bomb opponent at postincrement indirect bptr 
        add.i   #7, bptr      ; Increment bombing pointer by 7 (prime) to avoid pattern overlap and increase coverage
        jmn     bomb, bptr    ; Loop bombing until pointer wraps to zero

ptr     dat     #3, #0        ; Replication pointer starts at 3 (skip start code area)
bptr    dat     #12, #0       ; Bombing pointer starts at 12 to avoid overlap with replicator

        END start
