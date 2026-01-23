
;name polydwarfs
;author rodrigo setti
;strategy create 11 dwarfs in strategic positions with improved efficiency

org start

start   add.a   #800,   $7        ; adjust pointer positions for strategic bombing
        add.ab  #800,   $5
        add.ab  #800,   $3
        add.ab  #800,   $1

        mov.i   $5,     $809       ; copy dwarf code to target location
        mov.i   $5,     $809
        mov.i   $5,     $809
        spl.f   $806,   #0          ; spawn processes for each dwarf
        djn.b   $-8,    #11         ; loop to create 11 dwarfs (one more than original)

        spl.b   #2,    }0           ; advanced dwarf spawn, quicker splitting
        mov.i   $2,     }-1         ; move instructions to support dwarf movement
        dat.f   }-2,    }-2         ; data instruction to create bombing effect

end start
