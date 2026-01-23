
;name dwarf jumper improved
;author Rodrigo Setti
;strategy Rapidly launches many processes with post-increment indirect bombing for better distribution and tight looping for speed and survivability

ORG start

start   spl     1               ; create a new task with offset 1 for faster task creation
        mov.i   $1, >2         ; copy current instruction one forward for the jumper loop
        mov.i   $2, >3         ; copy bomb one step further to spread damage
        mov.i   #0, >3         ; bomb with post-increment indirect to spread damage and update pointers
        jmp     $-4            ; jump back to start for continuous splitting and bombing

        dat     #0, #0         ; bomb data to kill enemy processes

END start
