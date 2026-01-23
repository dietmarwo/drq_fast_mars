
;name dwarf avanÃ§ado melhorado 2
;author Rodrigo Setti (modificado para melhor desempenho)
;strategy Usa SPL para criar processos, usa pÃ³s-incremento para mover ponteiro, substitui DAT por JMP para nÃ£o suicidar, loop de bombardeio eficiente

        ORG     start

start   spl.b   #2, }0         ; Cria novo processo, incrementa alvo com pÃ³s-incremento B para prÃ³xima bomba
        mov.i   $2, }-1       ; Copia bomba para o endereÃ§o alvo, incrementa ponteiro com pÃ³s-incremento A
        jmp     start         ; Loop infinito para continuar bombardeando a memÃ³ria

        END     start
