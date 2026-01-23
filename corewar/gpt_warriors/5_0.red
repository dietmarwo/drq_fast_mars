
;name dwarf jumper improved
;author Rodrigo Setti
;strategy Rapidly launches many processes with post-increment indirect bombing for spread, quick looping and tight code for better speed and survivability

ORG start

start   spl     0               ; create a new task immediately (zero offset for fast split)
        mov.i   $1, >2         ; copy current instruction one step forward to continue jumper code
        mov.i   $2, >3         ; copy bomb one step further for spread
        mov.i   #0, >3         ; bomb with post-increment indirect on bomb pointer to spread damage and keep pointers updated
        jmp     $-4            ; jump back four to start splitting and bombing loop rapidly

        dat     #0, #0         ; bomb data to eliminate enemy processes

        END start
