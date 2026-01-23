;Name   Rato Replicante
;Author Rodrigo Setti
;Strat  MantÃ©m um processo ininterrupto de
;Strat  criaÃ§Ã£o de cÃ³pias funcionais.

org inicio

vetores dat.f   $0,         $2981       ;vetores de copia
inicio  mov.i   }vetores,   >vetores    ;copia instruÃ§Ã£o e incrementa vetores
        jmn.b   $inicio,    *vetores    ;loop de cÃ³pia -> enquanto nao encontrou um zero em B
        spl.b   <vetores,   {vetores    ;cria processo na cÃ³pia (ajuda a reestruturar ponteiro)
        add.x   #-31,       $-4         ;reestrutura ponteiros
        jmz.a   $inicio,    {vetores    ;loop do programa (ajuda a reestruturar ponteiro)
                                        ;ou se suicida se o programa estiver alterado
