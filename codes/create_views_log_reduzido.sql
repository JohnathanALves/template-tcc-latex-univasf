DELIMITER //
DROP PROCEDURE IF EXISTS create_views_log_reduzido;
CREATE PROCEDURE create_views_log_reduzido
(
	IN base_table VARCHAR(30)
)
BEGIN

	SET @base = base_table;

	SET @view_base_log_reduzido = CONCAT(@base, '_base_log_reduzido');

	SET @query_drop_base_log_reduzido = CONCAT(
		' DROP TABLE IF EXISTS ',
			@view_base_log_reduzido,
		' ;'
	);

	PREPARE stmt FROM @query_drop_base_log_reduzido;
	EXECUTE stmt;
	DEALLOCATE PREPARE stmt;

	SET @query_base_log_reduzido = CONCAT(
		'CREATE TABLE ',
		@view_base_log_reduzido,
		' AS SELECT @curRank := @curRank + 1 AS id,time,userid,course,module,action,ip,cmid FROM ',
		' mdl_log , (SELECT @curRank := 0) r ',
		' WHERE ',
			' action IN (\'login\' , \'view\', \'view forum\') ',
			' AND module IN ( ',
				' \'assign\', ',
				' \'forum\', ',
				' \'assignment\', ',
				' \'choice\', ',
				' \'feedback\', ',
				' \'survey\', ',
				' \'chat\', ',
				' \'quiz\', ',
				' \'resource\', ',
				' \'folder\', ',
				' \'url\', ',
				' \'page\', ',
				' \'book\', ',
				' \'user\'); '
	);

	PREPARE stmt FROM @query_base_log_reduzido;
	EXECUTE stmt;
	DEALLOCATE PREPARE stmt;

END //
DELIMITER ;
