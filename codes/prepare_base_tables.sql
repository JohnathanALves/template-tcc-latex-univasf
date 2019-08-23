DELIMITER //
DROP PROCEDURE IF EXISTS prepare_base_tables;
CREATE PROCEDURE prepare_base_tables()
BEGIN
	SET @adm = 'adm';
	SET @lic_pedagogia = 'lic_pedagogia';

	SELECT CONCAT('Criando tabela ', @adm);
	CALL create_base(@adm, '2013.2', 43);

	SELECT CONCAT('Criando tabela ', @lic_pedagogia);
	CALL create_base(@lic_pedagogia, '2014.2', 64);

	SELECT 'Criando Views compatilhadas...';
	CALL create_shared_views();

	SELECT CONCAT('Criando views ', @adm);
	CALL create_specific_views(@adm);

	SELECT CONCAT('Criando views ', @lic_pedagogia);
	CALL create_specific_views(@lic_pedagogia);
END //
DELIMITER ;
