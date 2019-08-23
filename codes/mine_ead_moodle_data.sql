DELIMITER //
DROP PROCEDURE IF EXISTS mine_ead_moodle_data;
CREATE PROCEDURE mine_ead_moodle_data()
BEGIN
	SET @adm_turma1 = 'adm_turma1';
	SET @lic_pedagogia = 'lic_pedagogia_turma1';
	SET @adm_turma2 = 'adm_turma2_old';
	SET @lic_pedagogia_turma2 = 'lic_pedagogia_turma2_old';

	SELECT CONCAT('Criando dataset ', @adm_turma1);
	CALL transational_distance(@adm_turma1);

	SELECT CONCAT('Criando dataset ', @lic_pedagogia);
	CALL transational_distance(@lic_pedagogia);

	SELECT CONCAT('Criando dataset ', @adm_turma2);
	CALL transational_distance(@adm_turma2);

	SELECT CONCAT('Criando dataset ', @lic_pedagogia_turma2);
	CALL transational_distance(@lic_pedagogia_turma2);
END //
DELIMITER ;
