DELIMITER //
DROP PROCEDURE IF EXISTS create_view_alunos;
CREATE PROCEDURE create_view_alunos()
BEGIN 

	DROP TABLE IF EXISTS alunos;
	CREATE TABLE alunos (
		disciplina_id BIGINT(10) NOT NULL,
		aluno_id BIGINT(10) NOT NULL,

		PRIMARY KEY(disciplina_id, aluno_id)
	) AS 
	SELECT c.id AS 'disciplina_id', u.id AS 'aluno_id'
	FROM mdl_role_assignments rs 
	INNER JOIN mdl_context e ON rs.contextid=e.id 
	INNER JOIN mdl_course c ON c.id = e.instanceid 
	INNER JOIN mdl_user u ON u.id=rs.userid 
	WHERE e.contextlevel=50 AND rs.roleid=5
	ORDER BY c.id, u.id;

END //
DELIMITER ;
