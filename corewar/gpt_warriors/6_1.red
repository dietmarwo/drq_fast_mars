
;name dwarf_mice
;author rodrigo setti
;strategy Improved dwarf: safer bombing range, efficient task splitting and clear self-modifying bombing loop

        ORG     start

start   MOV.I   $0, $-400         ; copy current instruction farther (400 steps back) for safer bombing
        ADD.I   #1, }-1          ; increment bomb address with A-number postincrement indirect addressing
        SEQ.I   }-1, #400        ; check if bomb address reached 400 steps (end)
        JMP.B   start            ; loop copying until done

        SPL     $-400            ; fork process at bomb start address to concurrently start bombing

loop    SPL     #3               ; split 3 new processes for faster replication and attack
        MOV.I   $2, }-1          ; copy bomb code with safe postincrement to prevent overwriting current bomb address
        DAT.F   }-2, }-2         ; place bomb (kills enemy)
        JMP     loop             ; repeat bombing loop continuously

        END     start
