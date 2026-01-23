
;name dwarf avanÃÂ§ado melhorado 2
;author Rodrigo Setti (modificado para melhor desempenho)
;strategy Usa SPL para criar processos, usa pÃÂ³s-incremento para mover ponteiro, substitui DAT por JMP para nÃÂ£o suicidar, loop de bombardeio eficiente

        ORG     start

start   spl.b   #2, }0         ; Cria novo processo, incrementa alvo com pÃÂ³s-incremento B para prÃÂ³xima bomba
        mov.i   $2, }-1       ; Copia bomba para o endereÃÂ§o alvo, incrementa ponteiro com pÃÂ³s-incremento A
        jmp     start         ; Loop infinito para continuar bombardeando a memÃÂ³ria

        END     start
