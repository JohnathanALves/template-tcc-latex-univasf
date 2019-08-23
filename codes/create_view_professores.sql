DELIMITER //
DROP PROCEDURE IF EXISTS create_view_professores;
CREATE PROCEDURE create_view_professores()
BEGIN

	DROP TABLE IF EXISTS professores;

	CREATE TABLE professores (
		disciplina_id BIGINT(10) NOT NULL,
		professor_id BIGINT(10) NOT NULL,
		PRIMARY KEY (disciplina_id, professor_id)
	) AS
	SELECT DISTINCT c.id AS 'disciplina_id', u.id AS 'professor_id'
	FROM mdl_role_assignments rs
	INNER JOIN mdl_context e ON rs.contextid=e.id
	INNER JOIN mdl_course c ON c.id = e.instanceid
	INNER JOIN mdl_user u ON u.id=rs.userid
	WHERE e.contextlevel=50 AND rs.roleid IN (3,4)
	ORDER BY c.id, u.id;

END //
DELIMITER ;
