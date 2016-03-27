//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiv�ve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Papai Attila
// Neptun : X6YVJM
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

const char *vertexSource = R"(
	#version 130
    precision highp float;

	uniform mat4 MVP;

	in vec2 vertexPosition;
	in vec3 vertexColor;
	out vec3 color;

	void main() {
		color = vertexColor;
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP;
	}
)";

const char *fragmentSource = R"(
	#version 130
    precision highp float;

	in vec3 color;
	out vec4 fragmentColor;

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
	vec4 operator+(const vec4& other) const
	{
		return vec4(v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2]);
	}
	vec4 operator-(const vec4& other) const
	{
		return vec4(v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2]);
	}
	vec4 operator*(const float num) const
	{
		return vec4(v[0] * num, v[1] * num, v[2] * num);
	}
	vec4 operator/(const float num) const
	{
		return vec4(v[0] / num, v[1] / num, v[2] / num);
	}
	float length()
	{
		return sqrt(v[0] * v[0] + v[1] * v[1]);
	}
};

struct Camera {
	float wCx, wCy;
	float wWx, wWy;
public:
	Camera() {
		Animate(0);
		wCx = 0;
		wCy = 0;
	}

	mat4 V() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() {
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() {
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wWx = 20;
		wWy = 20;
	}
};

Camera camera;

unsigned int shaderProgram;

class Triangle {
	unsigned int vao;
	float sx, sy;
	float wTx, wTy;
	float phi;
public:
	Triangle() {
		Animate(0, 0, 0, 0);
	}

	void Create(float r, float g, float b, float pMass) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo[2];
		glGenBuffers(2, &vbo[0]);


		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		float vertexCoords[] = { -0.1 * pMass, 0, 0.1 * pMass, 0, 0, 0.5 * pMass };
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(vertexCoords),
			vertexCoords,
			GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			2, GL_FLOAT,
			GL_FALSE,
			0, NULL);


		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		float vertexColors[] = { r, g, b, r, g, b, r, g, b };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);


		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Animate(float t, float pPhi, float pWTX, float pWTY) {
		sx = 1 * pow(sinf(t), 2) + 0.5f;
		sy = 1;
		wTx = pWTX;
		wTy = pWTY;
		phi = pPhi + t;
	}

	void Draw() {
		mat4 M(sx * cos(phi), -sx * sin(phi), 0, 0,
			-sy * -sin(phi), sy * cos(phi), 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1);

		mat4 MVPTransform = M * camera.V() * camera.P();

		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform);
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}
};


const int MAX_CTRL_POINT_COUNT = 20;
const int RESOLUTION = 12;
const int FLOAT_IN_VBO = 5;
const float TENSION = -0.8f;

struct CMSpline
{
	GLuint vao, vbo;
	float vertexData[(MAX_CTRL_POINT_COUNT)*FLOAT_IN_VBO*RESOLUTION];
	int nVertices;
	vec4 ctrlPoints[MAX_CTRL_POINT_COUNT];
	int nCtrlPoints;
	float ts[MAX_CTRL_POINT_COUNT];
	CMSpline()
	{
		nVertices = 0;
		nCtrlPoints = 0;
	}
	void Create()
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}
	void Draw() {
		if (nCtrlPoints > 2) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_LOOP, 0, nVertices);
		}
	}
	void AddPoint(float cX, float cY) {
		if (nCtrlPoints >= MAX_CTRL_POINT_COUNT) return;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();

		if (nCtrlPoints <= MAX_CTRL_POINT_COUNT)
		{
			AddCtrlPoint(wVertex.v[0], wVertex.v[1]);
			AddTrailingCtrlPoint();
		}
		else
		{
			return;
		}
		if (nCtrlPoints >= 4)
		{
			nVertices = 0;
			for (int i = 0; i < nCtrlPoints - 1; i++)
			{
				for (int j = 0; j < RESOLUTION; j++)
				{
					float deltaTime = ts[i + 1] - ts[i];
					vec4 point = r(ts[i] + j * deltaTime / RESOLUTION);
					AddVertexPoint(point.v[0], point.v[1]);
				}
			}
		}

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}
	vec4 r(float t)
	{
		for (int i = 0; i < nCtrlPoints - 1; i++)
		{
			if (ts[i] <= t && t <= ts[i + 1])
			{
				if (i == 0)
				{
					vec4 firstP = ctrlPoints[nCtrlPoints - 1];
					float firstT = ts[nCtrlPoints - 1];
					vec4 secondP = ctrlPoints[1];
					float secondT = ts[nCtrlPoints - 1] + (ts[1] - ts[0]);
					float time = ts[nCtrlPoints - 1] + t - ts[0];
					return Hermite(
						firstP,
						Velocity(TENSION, ctrlPoints[nCtrlPoints - 2], ts[nCtrlPoints - 2], ctrlPoints[nCtrlPoints - 1], ts[nCtrlPoints - 1], ctrlPoints[1], ts[nCtrlPoints - 1] + (ts[1] - ts[0])),
						firstT,
						secondP,
						Velocity(TENSION, ctrlPoints[nCtrlPoints - 1], ts[nCtrlPoints - 1], ctrlPoints[1], ts[nCtrlPoints - 1] + ts[1] - ts[0], ctrlPoints[2], ts[nCtrlPoints - 1] + ts[2] - ts[0]),
						secondT,
						time);
				}
				else if (i + 2 == nCtrlPoints)
				{
					return Hermite(
						ctrlPoints[nCtrlPoints - 2],
						Velocity(TENSION, ctrlPoints[nCtrlPoints - 3], ts[nCtrlPoints - 3], ctrlPoints[nCtrlPoints - 2], ts[nCtrlPoints - 2], ctrlPoints[nCtrlPoints - 1], ts[nCtrlPoints - 1]),
						ts[nCtrlPoints - 2],
						ctrlPoints[nCtrlPoints - 1],
						Velocity(TENSION, ctrlPoints[nCtrlPoints - 2], ts[nCtrlPoints - 2], ctrlPoints[nCtrlPoints - 1], ts[nCtrlPoints - 1], ctrlPoints[1], ts[nCtrlPoints - 1] + ts[1] - ts[0]),
						ts[nCtrlPoints - 1],
						t);
				}
				else
				{
					return Hermite(
						ctrlPoints[i],
						Velocity(TENSION, ctrlPoints[i - 1], ts[i - 1], ctrlPoints[i], ts[i], ctrlPoints[i + 1], ts[i + 1]),
						ts[i],
						ctrlPoints[i + 1],
						Velocity(TENSION, ctrlPoints[i], ts[i], ctrlPoints[i + 1], ts[i + 1], ctrlPoints[i + 2], ts[i + 2]),
						ts[i + 1],
						t);
				}
			}
		}
	}
	vec4 Hermite(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t)
	{
		vec4 a0 = p0;
		vec4 a1 = v0;
		vec4 a2 = (p1 - p0) * 3 / pow(t1 - t0, 2) - (v1 + v0 * 2) / (t1 - t0);
		vec4 a3 = (p0 - p1) * 2 / pow(t1 - t0, 3) + (v1 + v0) / pow(t1 - t0, 2);

		return a3 * pow(t - t0, 3) + a2 * pow(t - t0, 2) + a1 * (t - t0) + a0;
	}
	vec4 Velocity(float tension, vec4 r0, float t0, vec4 r1, float t1, vec4 r2, float t2)
	{
		vec4 velocity = ((r2 - r1) / (t2 - t1) + (r1 - r0) / (t1 - t0)) * ((1 - tension) / 2);
		return velocity;
	}
	void AddTrailingCtrlPoint()
	{
		if (nCtrlPoints == 1)
		{
			ctrlPoints[nCtrlPoints] = ctrlPoints[0];
			ts[nCtrlPoints] = ts[nCtrlPoints - 1] + 0.5f;
			nCtrlPoints++;
		}
		else
		{
			swapTrailingPoints();
		}
	}
	void swapTrailingPoints()
	{
		vec4 temp = ctrlPoints[nCtrlPoints - 2];
		ctrlPoints[nCtrlPoints - 2] = ctrlPoints[nCtrlPoints - 1];
		ctrlPoints[nCtrlPoints - 1] = temp;

		float time = ts[nCtrlPoints - 1];
		ts[nCtrlPoints - 2] = ts[nCtrlPoints - 1];
		ts[nCtrlPoints - 1] = time + 0.5f;
	}
	void AddCtrlPoint(float wX, float wY)
	{
		ctrlPoints[nCtrlPoints].v[0] = wX;
		ctrlPoints[nCtrlPoints].v[1] = wY;

		ts[nCtrlPoints] = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

		nCtrlPoints++;
	}
	void AddVertexPoint(float wX, float wY)
	{
		vertexData[5 * nVertices] = wX;
		vertexData[5 * nVertices + 1] = wY;
		vertexData[5 * nVertices + 2] = 1;
		vertexData[5 * nVertices + 3] = 1;
		vertexData[5 * nVertices + 4] = 0;
		nVertices++;
	}
};

const float STAR_ROAD_TRIP_TIME = 5.0f;
const float GRAVITATIONAL_CONSTANT = 0.15f;
const float FRICTION = 0.8f;

struct Star
{
	Triangle parts[20];
	CMSpline *spline;
	Star *mainStar;
	bool isActive;
	float mass;
	vec4 position;
	vec4 velocity;
	int verticesCount;
	float prevTime;

	Star(CMSpline *pSpline, Star* pMainStar, int pVerticesCount, float pMass)
	{
		spline = pSpline;
		mainStar = pMainStar;
		mass = pMass;
		isActive = false;
		verticesCount = pVerticesCount;
		prevTime = 0;
	}
	void Create(float r, float g, float b)
	{
		for (int i = 0; i < verticesCount; i++)
		{
			parts[i].Create(r, g, b, mass);
		}
	}
	void Animate(float t)
	{
		if (spline)
		{
			if (spline->nCtrlPoints >= 4)
			{
				float deltaDegree = 360.0f / verticesCount;
				for (int i = 0; i < verticesCount; i++)
				{
					float phi = (i * deltaDegree) * M_PI / 180;
					float time = getRelativeTime();
					position = spline->r(spline->ts[0] + time);
					parts[i].Animate(t, phi, position.v[0], position.v[1]);
				}
			}
		}
		else if (mainStar)
		{
			if (mainStar->isActive)
			{
				if (prevTime == 0.0f) prevTime = t;
				float dt = t - prevTime;
				prevTime = t;
				float deltaDegree = 360.0f / verticesCount;
				for (int i = 0; i < verticesCount; i++)
				{
					float phi = (i * deltaDegree) * M_PI / 180;

					vec4 force = getForceBetweenMasses(mainStar);
					vec4 forceFriction = velocity * FRICTION;
					force = force - forceFriction;
					vec4 acceleration = force / mass;
					velocity = velocity + acceleration *dt;
					position = position + velocity * dt;
					parts[i].Animate(t, phi, position.v[0], position.v[1]);
				}
			}
		}
	}
	void Draw()
	{
		if (spline)
		{
			if (spline->nCtrlPoints >= 4)
			{
				isActive = true;
				for (int i = 0; i < verticesCount; i++)
				{
					parts[i].Draw();
				}
			}
		}
		else if (mainStar)
		{
			if (mainStar->isActive)
			{
				for (int i = 0; i < verticesCount; i++)
				{
					parts[i].Draw();
				}
			}
		}
	}
	// forras:
	// https://en.wikipedia.org/wiki/Newton's_law_of_universal_gravitation#Vector_form
	vec4 getForceBetweenMasses(Star* other)
	{
		float G = GRAVITATIONAL_CONSTANT;
		float distance = sqrt(pow(other->position.v[0] - position.v[0], 2) + pow(other->position.v[1] - position.v[1], 2));
		// forrastol elteres: ha nagyon kicsi a tavolsag, akkor 1-nek allitom be a tavolsagot
		if (distance < 0.5f) distance = 1;
		vec4 unitVector = (other->position - position) / distance;
		float numerator = mass * other->mass;
		float denominator = pow(distance, 2);
		return unitVector * numerator / denominator * G;
	}
	float getRelativeTime()
	{
		float segmentTotalTime = spline->ts[spline->nCtrlPoints - 1] - spline->ts[0];
		float relTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
		while (relTime >= STAR_ROAD_TRIP_TIME)
		{
			relTime -= STAR_ROAD_TRIP_TIME;
		}
		relTime = segmentTotalTime / STAR_ROAD_TRIP_TIME * relTime;
		return relTime;
	}
};

void updateCameraCoords(Camera *cam, Star *star)
{
	cam->wCx = star->position.v[0];
	cam->wCy = star->position.v[1];
}

CMSpline lineStrip;
Star Polaris(&lineStrip, 0, 8, 7);
Star Sirius(0, &Polaris, 9, 5);
Star Rigel(0, &Polaris, 10, 4);
bool isCameraFollowingStar = false;


void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	lineStrip.Create();
	Polaris.Create(1, 1, 0);
	Sirius.Create(1, 1, 1);
	Sirius.position = vec4(2, 2);
	Rigel.Create(0.5f, 0.6f, 0.7f);
	Rigel.position = vec4(-8, -9);

	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	glBindAttribLocation(shaderProgram, 0, "vertexPosition");
	glBindAttribLocation(shaderProgram, 1, "vertexColor");


	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");

	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	lineStrip.Draw();
	Polaris.Draw();
	Sirius.Draw();
	Rigel.Draw();
	glutSwapBuffers();
}


void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();
	else if (key == 32) isCameraFollowingStar = true;
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;
		lineStrip.AddPoint(cX, cY);
		glutPostRedisplay();
	}
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	float sec = time / 1000.0f;
	if (isCameraFollowingStar)
	{
		updateCameraCoords(&camera, &Polaris);
	}
	camera.Animate(sec);
	Polaris.Animate(sec);
	Sirius.Animate(sec);
	Rigel.Animate(sec);
	glutPostRedisplay();
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

