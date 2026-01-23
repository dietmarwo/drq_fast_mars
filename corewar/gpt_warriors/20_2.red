
;name MirrorBot Improved Optimized v3
;author CoreWarAssistant
;strategy Interleaved replicator and bomber with refined pointer wrapping using postincrement
;         More efficient replication and bombing with ensured wrapping and uniform bombing spread
;         Using postincrement indirect for better pointer increment timing and avoiding missed sites

        ORG start

start   spl     copy          ; Start replicator process
        spl     bomb          ; Start bomber process in parallel
        jmp     loop          ; Main monitoring loop

copy    mov.i   0, }ptr       ; Replicate current instruction to address pointed by postincrement indirect ptr (use pointer, then increment)
        jmn     copy, ptr     ; Loop until pointer wraps to zero (non-zero jump), replication complete

loop    jmz     start, ptr    ; If replication pointer wrapped to zero, restart processes (respawn)
        jmp     loop          ; Otherwise continue monitoring

bomb    mov.i   bomb, }bptr   ; Bomb opponent at postincrement indirect bptr (use pointer, then increment)
        add.i   #4, bptr      ; Increment bombing pointer by 4 to spread bombs evenly
        jmn     bomb, bptr    ; Loop bombing until pointer wraps to zero

ptr     dat     #3, #0        ; Replication pointer starts at 3 (skip start code area)
bptr    dat     #12, #0       ; Bombing pointer starts at 12 to avoid overlap with replicator

        END start
