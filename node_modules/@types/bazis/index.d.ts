declare interface System {
    /**
     * Вывести диагностическое сообщение (для отладки?)
     * @param str сообщение
     */
    log(str: string);
    /**
     * Подключить указанный файл
     */
    include(filename: string);
    /**
     * Возвращает true, если файл существует
     * @deprecated используйте fs.existsSync
     * @param filename путь к файлу
     */
    fileExists(filename: string): boolean;
    /**
     * Открыть диалог выбора файла
     * @param extension расширение файла
     */
    askFileName(extension: string): string;
    /**
     * Открыть диалог выбора папки
     * @param caption заголовок папки
     * @param defaultFolder Папка по умолчанию
     */
    askFolder(caption?: string, defaultFolder?: string): string;
    /**
     * Получить имя файла без пути
     */
    getFileName(filename: string): string;
    /**
     * Получить имя файла без пути и расширения
     */
    getFileNameWithoutExtension(filename: string): string;
    /**
     * Открыть диалог сохранения файла
     * @param ext расширение файла
     */
    askFileNameSave(ext: string): string;
    /**
     * Записать текст в файл
     * @deprecated используйте fs.writeFileSync
     * @param filename путь к файлу
     * @param  content содержимое файла
     */
    writeTextFile(filename: string, content: string);
    /**
     * Записать текст в файл с запросом имени файла
     * @param extension Расширение файла
     * @param  content содержимое
     */
    askWriteTextFile(extension: string, content: string);
    /**
     * Считать текст из файла
     * @param filename Имя файла
     */
    readTextFile(filename: string): string;
    /**
     * Считать текст из файла с запросом выбора файла
     * @param extension Расширение файла
     */
    askReadTextFile(extension: string): string;
    /**
     * Выполнить зашифрованный код
     * @param str код скрипта
     */
    secureExec(str: string);
    /**
     * Выполнить внешнюю программу
     * @param str код программы
     */
    exec(str: string): boolean;
    /**
     * Задерживает выполнение программы на указанное время
     * @param ms время (в миллисекундах)
     */
    sleep(ms: number);
    /**
     * Текущая версия Bazis API
     */
    apiVersion: number;
    /**
     *
     */
    developerApiVersion: number;
}

declare interface IDispatch {
    /**
     * Информациях о методах и свойствах объекта
     */
    GetTypeInfo: string;

}

declare interface Model3D extends List3D {
    /**
     * Размер модели
     */
    GSize: Vector;
    /**
     * Габарит модели
     */
    GMin: Vector;
    /**
     * Габарит модели
     */
    GMax: Vector;
    /**
     * Выделенный объект модели
     */
    Selected: Object3;
    /**
     * Количество выделенных элементов
     */
    SelectionCount: number;
    /**
     * Список выделенных элементов
     */
    Selections: Array<Object3>;
    /**
     * Количество объектов в модели
     */
    Count: number;
    /**
     * Список объектов модели
     */
    Objects: Array<Object3>;
    /**
     * Выполнить функцию для каждой панели
     */
    forEachPanel(callbackfn: (panel: Panel) => void): void;
    /**
     * Выполнить функцию для каждого объекта модели
     */
    forEach(callbackfn: (obj: Object3) => void): void;
    /**
     * Снять выделение со всех объектов
     */
    UnSelectAll(): void;
}

declare interface Action3D {
    /**
     * Если false, то запрещены любые функции взаимодействия с пользователем
     */
    Interactive: boolean;
    /**
     * Продолжить вызывать прерывания по завершению основного тела скрипта, не завершая команды
     */
    Continue();
    /**
     * Применить изменения в модели внесенные в скрипте
     */
    Commit();
    /**
     * Отменить изменения в модели внесенные в скрипте
     */
    Revert();
    /**
     * Завершить команду
     */
    Finish();
    /**
     * Отменить команду
     */
    Cancel();
    /**
     * Выполнить функцию, в которой доступны запросы Get*
     * @param func функция для выполнения
     */
    AsyncExec(func: Function);
    /**
     * Позиция курсора X
     */
    MouseX: number;
    /**
     * Позиция курсора Y
     */
    MouseY: number;
    /**
     * Текущая позиция маркера
     */
    Pos3: Vector;
    /**
     * Нормаль к текущему виду
     */
    ViewDir: Vector;
    /**
     * Вектор вверх текущего вида
     */
    UpDir: Vector;
    /**
     * Вектор вправо текущего вида
     */
    RightDir: Vector;
    /**
     * Разрешать подсвечивать точки
     */
    ShowPoints: boolean;
    /**
     * Разрешать подсвечивать ребра
     */
    ShowEdges: boolean;
    /**
     * Установить строку подсказки
     */
    Hint: string;
    /**
     * Установить сообщение об ошибке
     */
    ErrorHint: string;
    /**
     * Установить мигающую подсказку
     */
    BlinkHint: string;
    /**
     * Включить режим Орто относительно точки
     * @param pos Координаты точки
     */
    BeginOrtho3(pos: Vector);
    /**
     * Отключить режим Орто
     */
    EndOrtho3();
    /**
     * Сдвинуть курсор к ближайшей точке привязки
     */
    CursorToClosestPoint();
    /**
     * Сдвинуть курсор к ближайшей линии
     */
    CursorToClosestLine();
    /**
     * Сдвинуть курсор к ближайшей середине линии
     */
    CursorToMiddleOfLine();
    /**
     * Найти точку на модели в текущем положении курсора мыши
     */
    Find3DPoint(): Vector;
    /**
     * Найти точку на плоскости X0Z
     */
    Find3DPointXZPlane(): Vector;
    /**
     * Найти объект под курсором мыши
     */
    Get3DObject(): Object3;
    /**
     * Обработчик щелчка мыши
     */
    OnClick: Function;
    /**
     * Обработчик щелчка мыши
     */
    OnMove: Function;
    /**
     * Обработчик начала работы скрипта. Вызывается после загрузки значений свойств
     */
    OnStart: Function;
    /**
     * Обработчик завершения работы скрипта
     */
    OnFinish: Function;
    /**
     * Запустить тонирование изображения
     */
    RayTraceScene();
    /**
     * Обработчик завершения тонирования изображения
     */
    OnRayTraceFinished: Function;
    /**
     * Набор редактируемых свойств
     */
    Properties: RootProperties;
    /**
     * Загрузить модель из файла
     */
    LoadModel(filename: string): boolean;
    /**
     * Сохранить модель в файл
     */
    SaveModel(filename: string);
    /**
     * Создать новую модель
     */
    NewModel();
    /**
     * Создать новую фурнитуру
     */
    NewFurniture();
    /**
     * Создать новый фрагмент
     */
    NewFragment();
    /**
     * Имя файла текущей модели
     */
    ModelFilename: string;
    /**
     * Расстановка позиций
     * @param Mode режим расстановки
     * @param Model структурный объект
     */
    ArrangePositions(Mode: FurnPositionMode, Model: List3D): boolean;
    /**
     * Замена фурнитуры
     * @param Old список названий старой фурнитуры
     * @param New список значений новой фурнитуры
     * @param Fasteners список объектов для замены
     */
    ReplaceFurniture(Old: string[], New: InfFurniture[], Fasteners: Object3[]);
    /**
     * Вызов окна выбора материала из базы. Возвращает полное имя материала
     */
    ChooseMaterial(): string;
}
/**
 * Артикул модели
 */
declare interface FurnArticle {
    /**
     * Имя модели
     */
    Name: string
    NameWithCode(): string;
    Code: string;
    FurnType: string;
    Author: string;
    Subject: string;
    Enterprise: string;
    Notes: string;
    /**
     * Код заказа
     */
    OrderCode: string;
    /**
     * Имя заказа
     */
    OrderName: string;
    /**
     * Тип установки фурнитуры/фрагмента
     */
    DatumMode: DatumMode;
}

declare interface ScriptMenu {
    /**
     * Имя свойства
     */
    Name: string;
    /**
     * Обработчик изменения свойства и вложенных свойств
     */
    OnChange: Function;
    /**
     * Количество вложенных свойств
     */
    Count: number;
    /**
     * Список вложенных свойств
     */
    Items: Array<ScriptProperty>;
    /**
     * Очистить вложенные свойства
     */
    Clear();
    /**
     * Флаг, сохраняется ли свойство в файл
     */
    Store: boolean;
    /**
     * Сохранить введенные пользователем данные в файле xml
     * @param filename
     */
    Save(filename: string);
    /**
     * Загрузить значения полей из файла xml
     * @param filename
     */
    Load(filename: string): boolean;
    /**
     * Создать вложенную группу свойств
     * @param caption Название группы
     */
    NewGroup(caption: string): ScriptGroupProperty;
    /**
     * Создать строковое свойство
     * @param caption Название свойства
     * @param value
     */
    NewString(caption: string, value?: string): ScriptStringProperty;
    /**
     * Создать свойство вида Да/Нет
     * @param caption Название свойства
     */
    NewBool(caption: string, value?: boolean): ScriptBooleanProperty;
    /**
     * Создать числовое свойство
     * @param caption Название свойства
     */
    NewNumber(caption: string, value?: number): ScriptNumberProperty;
    /**
     * Создать кнопку
     * @param caption Название свойства
     */
    NewButton(caption: string): ScriptButtonProperty;

}

declare interface PropertyLayout {
    /**
     * Левая граница
     */
    Left: number;
    /**
     * Правая граница
     */
    Right: number;
    /**
     * Верхняя граница
     */
    Top: number;
    /**
     * Нижняя граница
     */
    Bottom: number;
    /**
     * Ширина свойства
     */
    Width: number;
    /**
     * Высота свойства
     */
    Height: number;

}

declare interface ScriptProperty {
    /**
     * Имя свойства
     */
    Name: string;
    /**
     * Родительское свойство
     */
    Owner: ScriptProperty;
    /**
     * Разрешить редактирование имени (для создания таблиц)
     */
    NameEditable: boolean;
    /**
     * Возможность выделение свойства пользователем
     */
    Enabled: boolean;
    /**
     * Выделение вложенных свойств
     */
    ChildrenEnabled: boolean;
    /**
     * Видимость свойства в окне свойств
     */
    Visible: boolean;
    /**
     * Развернуты ли вложенные свойства
     */
    Expanded: boolean;
    /**
     * Обработчик изменения свойства и вложенных свойств
     */
    OnChange: Function;
    /**
     * Обработчик изменения свойства
     */
    OnValueChange: Function;
    /**
     * Обработчик активации свойства или меню
     */
    OnActivate: Function;
    /**
     * Обработчик деактивации свойства или меню
     */
    OnDeactivate: Function;
    /**
     * Цвет фона
     */
    BackColor: number;
    /**
     * Пользовательское число
     */
    Tag: number;
    /**
     * Количество вложенных свойств
     */
    Count: number;
    /**
     * Список вложенных свойств
     */
    Items: Array<ScriptProperty>;
    /**
     * Очистить вложенные свойства
     */
    Clear();
    /**
     * Флаг, сохраняется ли свойство в файл
     */
    Store: boolean;
    /**
     * Сохранить введенные пользователем данные в файле xml
     * @param filename
     */
    Save(filename: string);
    /**
     * Загрузить значения полей из файла xml
     * @param filename
     */
    Load(filename: string): boolean;
    /**
     * Проверка корректности значения
     */
    OnValueValidate: Function;
    /**
     * Флаг корректности введенного значения, выставляется пользовательским обработчиком
     */
    ValueValid: boolean;
    /**
     * Проверить значение свойства и вложенных свойств
     */
    Validate(): boolean;
    /**
     * Расположение свойства на форме (для форм)
     */
    Layout: PropertyLayout;
    /**
     * Задать расположение свойства на форме
     * @param  L Отступ от левого края родительского объекта
     * @param  T Отступ от верхнего края родительского объекта
     * @param  W Ширина
     * @param  H Высота
     */
    SetLayout(L: number, T: number, W: number, H: number);
    /**
     * Задать выравнивание компонента (для форм)
     */
    Align: AlignType;
    /**
     * Выравнивать с отступами (для форм)
     */
    AlignWithMargins: boolean;
    /**
     * Отступы между компонентами (для форм)
     * @param  L Отступ слева
     * @param  R Отступ справа
     * @param  T Отступ сверху
     * @param  B Отступ снизу
     */
    SetMargins(L: number, R: number, T: number, B: number);
    /**
     * Выравнивание текста в надписи (для форм)
     */
    Alignment: AlignmentType;
    /**
     * Создать вложенную группу свойств
     * @param caption Название группы
     */
    NewGroup(caption: string): ScriptGroupProperty;
    /**
     * Создать вложенную группу свойств c рисунком
     * @param caption Название свойства
     * @param imagefile путь к файлу с рисунком
     */
    NewImage(caption: string, imagefile: string): ScriptGroupProperty;
    /**
     * Создать строковое свойство
     * @param caption Название свойства
     */
    NewString(caption: string): ScriptStringProperty;
    /**
     * Создать свойство вида Да/Нет
     * @param caption Название свойства
     */
    NewBool(caption: string): ScriptBooleanProperty;
    /**
     * Создать числовое свойство
     * @param caption Название свойства
     */
    NewNumber(caption: string): ScriptNumberProperty;
    /**
     * Создать кнопку
     * @param caption Название свойства
     */
    NewButton(caption: string): ScriptButtonProperty;
    /**
     * Создать свойство с кнопкой редактирования
     * @param caption Название свойства
     */
    NewSelector(caption: string): ScriptSelectorProperty;
    /**
     * Создать свойство - выпадающий список
     * @param caption Название свойства
     * @param  Item1 Первый элемент списка
     */
    NewCombo(caption: string, Item1?: string): ScriptComboProperty;
    /**
     * Создать свойство типа материал
     * @param caption Название свойства
     */
    NewMaterial(caption: string): ScriptMaterialProperty;
    /**
     * Создать свойство типа материал
     * @param caption Название свойства
     */
    NewButt(caption: string): ScriptButtProperty;
    /**
     * Создать свойство типа материал
     * @param caption Название свойства
     */
    NewFurniture(caption: string): ScriptFurnitureProperty;
    /**
     * Создать свойство типа цвет
     * @param caption Название свойства
     */
    NewColor(caption: string): ScriptColorProperty;
    /**
     * Создать разделитель
     */
    NewSeparator(): ScriptProperty;
    /**
     * Создать надпись
     * @param caption Название свойства
     */
    NewLabel(caption: string): ScriptProperty;
    /**
     * Всплывающее меню
     */
    PopupMenu: ScriptMenu;
    /**
     * Выпадающее меню
     */
    DropDownMenu: ScriptMenu;

}

declare interface RootProperties extends ScriptProperty {
    /**
     * Создать объект с информацией о фурнитуре
     */
    NewFurnitureValue(): InfFurniture;
}

/**
 * Тип выравнивания текста на надписи
 */
declare enum AlignmentType {
    /**
     * Выравнивание по левому краю
     */
    Left,
    /**
     * Выравнивание по правому краю
     */
    Right,
    /**
     * Выравнивание по центру
     */
    Center

}

/**
 * Типы выравнивания компонентов на форме
 */
declare enum AlignType {
    /**
     * Не выравнивается по форме
     */
    None,
    /**
     * Выравнивается по верхнему краю
     */
    Top,
    /**
     * Выравнивается по нижнему краю
     */
    Bottom,
    /**
     * Выравнивается по левому краю
     */
    Left,
    /**
     * Выравнивается по правому краю
     */
    Right,
    /**
     * Выравнивается по всей площади родительского компонента
     */
    Client

}

/**
 * Позиция немодального окна
 */
declare enum WindowPosition {
    /**
     * Стандартная позиция формы
     */
    Default,
    /**
     * Форма пристыковывается слева
     */
    Left,
    /**
     * Форма пристыковывается справа
     */
    Right

}

/**
 * Позиция фурнитуры при установке крепежа по схеме
 *
 */
declare enum FurniturePosition {
    /**
     * Установка фурнитуры внутри стыка
     */
    Inside,
    /**
     * Установка фурнитуры снаружи стыка
     */
    Outside,
    /**
     * Установка фурнитуры вверху стыка (только для стыков с горизонтальной панелью)
     */
    Up,
    /**
     * Установка фурнитуры внизу стыка (только для стыков с горизонтальной панелью)
     */
    Down

}


/**
 * Тип ошибки анализа модели
 */
declare enum ErrorType {
    /**
     * Пересечение объектов
     */
    ObjIntersection,
    /**
     * Пересечение фурнитуры
     */
    FastIntersection,
    /**
     * Неправильная установка фурнитуры
     */
    FastIncorrect,
    /**
     * Материала нет в наличии
     */
    MatNotExists,
    /**
     * Материал отсутствует на складе
     */
    MatOutOfStock,
    /**
     * Панель невозможно разместить на плите
     */
    PanelTooLarge,
    /**
     * Пластик невозможно разместить на панели
     */
    PlasticTooLarge,
    /**
     * Панель не закреплена
     */
    PanelNotFixed

}
/**
 * Параметры обнаруженной ошибки
 */
declare interface InspectorError {
    /**
     * Тип ошибки
     */
    ErrorType: ErrorType;
    /**
     * Количество объектов, относящихся к ошибке
     */
    ErrorObjectsCount: number;
    /**
     * Список объектов, относящихся к ошибке
     */
    ErrorObjects: Array<Object3>;
    /**
     * Сообщение ошибки
     */
    ErrorMessage: string;
    /**
     * Имена объектов в ошибке
     */
    ObjectsNames: string;

}
/**
 * Режим расстановки позиций
 */
declare enum FurnPositionMode {
    /**
     * Раставить позиции заново
     */
    All = 0,
    /**
     * Раставить позиции у новых объектов
     */
    New = 1,
    /**
     * Проверить позиции
     */
    Check = 2
}

/**
 * Параметры анализа модели
 */
declare interface InspectorOptions {
    /**
     * Проверка пересечения объектов
     */
    ObjIntersectionAnalyze: boolean;
    /**
     * Проверка пересечения фурнитуры
     */
    FastIntersectionAnalyze: boolean;
    /**
     * Проверка корректности фурнитуры
     */
    FastIncorrectAnalyze: boolean;
    /**
     * Проверка скрепления панелей
     */
    PanelNotFixedAnalyze: boolean;
    /**
     * Проверка размера панели на плите
     */
    PanelTooLargeAnalyze: boolean;
    /**
     * Проверка размера пластика на панели
     */
    PlasticTooLargeAnalyze: boolean;
    /**
     * Проверка материала в наличии
     */
    MatNotExistsAnalyze: boolean;
    /**
     * Проверка материала на складе
     */
    MatOutOfStockAnalyze: boolean;

}
/**
 * Тип монтирования фурнитуры/фрагмента
 */
declare enum DatumMode {
    /**
     * Отсутствует
     */
    None,
    /**
     * На плоскость
     */
    Face,
    /**
     * по двум непараллельным плоскостям
     */
    FaceFace,
    /**
     * по плоскости и середине торца панели
     */
    FaceButt,
    /**
     * По плоскости и ребру
     */
    FaceEdge,
    /**
     * на 2 параллельные плоскости
     */
    ParallelFaces,
    /**
     * секция
     */
    Box,
    /**
     * Крепеж по схеме
     */
    Scheme
}

declare interface ModelInspector {
    /**
     * Проверить модель
     * @param Model
     */
    Run(Model: List3D);
    /**
     * Список ошибок
     */
    ErrorList: Array<InspectorError>;
    /**
     * Опции анализа
     */
    Options: InspectorOptions;

}

declare interface ScriptForm {
    /**
     * Набор редактируемых свойств
     */
    Properties: ScriptProperty;
    /**
     * Показать форму
     * @param WindowPos
     */
    Show(WindowPos?: WindowPosition);
    /**
     * Показать модальную форму
     */
    ShowModal(): boolean;
    /**
     * Заголовок формы
     */
    Caption: string;
    /**
     * Ширина формы
     */
    Width: number;
    /**
     * Высота формы
     */
    Height: number;
    /**
     * Минимальная ширина формы
     */
    MinWidth: number;
    /**
     * Минимальная высота формы
     */
    MinHeight: number;
    /**
     * Видимость формы
     */
    Visible: boolean;
    /**
     * Положение левого края формы
     */
    Left: number;
    /**
     * Положение верхнего края формы
     */
    Top: number;
    /**
     * Показывать кнопку "ОК" на форме
     */
    OKButton: boolean;
    /**
     * Текст кнопки "ОК"
     */
    OKButtonCaption: string;
    /**
     * Показывать кнопку "Отмена" на форме
     */
    CancelButton: boolean;
    /**
     * Текст кнопки "Отмена"
     */
    CancelButtonCaption: string;
    /**
     * Обработчик закрытия формы
     */
    OnClose: Function;
    /**
     * Закрыть форму
     */
    Close();
    /**
     * Обработчик открытия формы
     */
    OnShow: Function;
    /**
     * Возможность изменять размеры формы
     */
    Resizable: boolean;
    /**
     * Возможность пристыковывать не модальную форму
     */
    Dockable: boolean;
    /**
     * Обработчик нажатия на кнопку OK
     */
    OnOkButtonClick: Function;
    /**
     * Обработчик нажатия на кнопку Cancel
     */
    OnCancelButtonClick: Function;

}

declare interface ScriptParamFastenerDB {
    /**
     * Загрузить базу из файла
     * @param  filename
     */
    LoadFromFile(filename: string): string;
    /**
     * Добавить базу из файла
     * @param  filename
     */
    AddFromFile(filename: string): string;
    /**
     * Сохранить базу в файл
     * @param  filename
     */
    SaveToFile(filename: string): string;

}

declare interface ScriptGroupProperty extends ScriptProperty {
    /**
     * Путь к изображению
     */
    Image: string;
    /**
     * Максимальная высота
     */
    MaxHeight: number;
    /**
     * Добавление полосы прокрутки
     */
    Scrollable: boolean;
}

declare interface ScriptStringProperty extends ScriptProperty {
    /**
     * Значение свойства
     */
    Value: string;

}

declare interface ScriptBooleanProperty extends ScriptProperty {
    /**
     * Значение свойства
     */
    Value: boolean;

}

declare interface ScriptNumberProperty extends ScriptProperty {
    /**
     * Минимальное значение свойства
     */
    MinValue: number;
    /**
     * Максимальное значение свойства
     */
    MaxValue: number;
    /**
     * Значение свойства
     */
    Value: number;
    /**
     * Шаг изменения значения (при заданных минимальном и максимальном значениях)
     */
    ValueStep: number;

}

declare interface ScriptButtonProperty extends ScriptProperty {
    /**
     * Обработчик нажатия на кнопку
     */
    OnClick: Function;

}

declare interface ScriptSelectorProperty extends ScriptProperty {
    /**
     * Значение свойства
     */
    Value: string;
    /**
     * Обработчик нажатия на кнопку редактирования свойства
     */
    OnClick: Function;

}

declare interface ScriptComboProperty extends ScriptProperty {
    /**
     * Индекс текущего значения в списке
     */
    ItemIndex: number;
    /**
     * Добавить элемент в список
     * @param item
     */
    AddItem(item: string);
    /**
     * Текущее значение в списке
     */
    Value: string;

}

declare interface ScriptMaterialProperty extends ScriptProperty {
    /**
     * Толщина материала
     */
    Thickness: number;
    /**
     * Ширина материала
     */
    Width: number;
    /**
     * Установить активным. Все последующие элементы будут построены из этого материала
     */
    SetActive();

}

declare interface ScriptButtProperty extends ScriptProperty {
    /**
     * Толщина кромки
     */
    Thickness: number;
    /**
     * Ширина ленты
     */
    Width: number;
    /**
     * Установить активным. Все последующие элементы будут построены из этого материала
     */
    SetActive();

}

declare interface ScriptFurnitureProperty extends ScriptProperty {
    /**
     * Значение свойства
     */
    Value: InfFurniture;
    /**
     * Очистить значение
     */
    ClearValue(): void;
}

declare interface ScriptColorProperty extends ScriptProperty {
    /**
     * Значение свойства
     */
    Value: number;

}

declare interface Undo3D {
    /**
     * Начать запись изменений объекта
     * @param obj
     */
    Changing(obj: Object3);
    /**
     * Начать запись изменений объекта и его составляющих
     * @param obj
     */
    RecursiveChanging(obj: Object3);

}

declare interface FurnMaterial {
    /**
     * Наименование материала
     */
    Name: string;
    /**
     * Толщина материала
     */
    Thickness: number;
    /**
     * Ширина материала
     */
    Width: number;
    /**
     * Создать материал из наименования и толщины (ширины)
     * @param name Наименование материала
     * @param thick Толщина материала
     */
    Make(name: string, thick: number);

}

declare interface Vector {
    /**
     * Координата X
     */
    x: number;
    /**
     * Координата Y
     */
    y: number;
    /**
     * Координата Z
     */
    z: number;

}

declare interface Point {
    /**
     * Координата Х
     */
    x: number;
    /**
     * Координата Y
     */
    y: number;

}

declare interface Edge3 {
    /**
     * Начало ребра в ЛСК
     */
    First: Vector;
    /**
     * Конец ребра в ЛСК
     */
    Last: Vector;
    /**
     * Начало ребра
     */
    GFirst: Vector;
    /**
     * Конец ребра
     */
    GLast: Vector;

}

declare interface RayInfo{
    /** Положение луча */
    Position: Vector;
    /** Направление луча */
    Direction: Vector;
    /** Рассттояние до обнаруженного пересечения (не превышает заданного перед проверкой пересечений) */
    Distance: number;
    /** Прекратить поиск пересечений луча с объектами после нахождения первого пересечения*/
    FindFirstIntersection: boolean;
    /** Проверять пересечения с поверхностями, если луч входит изнутри */
    IntersectBackFaces: boolean;
    /** Проверять пересечения с невидимыми объектами */
    IntersectHiddenObjects: boolean;
    /** Обнулить информацию о пересечении с объектом */
    Start();

    /** Флаг - найдено ли пересечение */
    Intersected: boolean;
    /** Найденная точка пересечения*/
    InterPos: Vector;
    /** Объект, с которым найдено пересечение */
    Object3D: Object3;
}

declare interface Object3 extends Object {
    /**
     * Наименование
     */
    Name: string;
    /**
     * Артикул
     */
    ArtPos: string;
    /**
     * Родитель объекта
     */
    Owner: List3D;
    /**
     * Видимость объекта
     */
    Visible: boolean;
    /**
     * Является ли объект выделенным
     */
    Selected: boolean;
    /**
     * Цвет линий объекта
     */
    Color: number;
    /**
     * Является ли объект структурным
     */
    List: boolean;
    /**
     * Привести объект к структурному
     */
    AsList(): List3D;
    /**
     * Привести объект к типу панели
     */
    AsPanel: Panel;
    /**
     * Положение объекта
     */
    Position: Vector;
    /**
     * Координата x
     */
    PositionX: number;
    /**
     * Координата y
     */
    PositionY: number;
    /**
     * Координата z
     */
    PositionZ: number;
    /**
     * Установить нулевые положение и ориентацию объекта
     */
    SetDefaultTransform();
    /**
     * Сместить объект
     * @param dir Вектор смещения
     */
    Translate(dir: Vector);
    /**
     * Повернуть вокруг заданной оси
     * @param axis Ось поворота
     * @param angle Угол (в градусах)
     */
    Rotate(axis: Vector, angle: number);
    /**
     * Переместить объект относительно глобальной системы координат
     * @param dir Вектор смещения
     */
    TranslateGCS(dir: Vector);
    /**
     * Повернуть объект относительно глобальной системы координат
     * @param axis Ось поворота
     * @param angle Угол (в градусах)
     */
    RotateGCS(axis: Vector, angle: number);
    /**
     * Повернуть вокруг оси X
     * @param angle Угол (в градусах)
     */
    RotateX(angle: number);
    /**
     * Повернуть вокруг оси Y
     * @param angle Угол (в градусах)
     */
    RotateY(angle: number);
    /**
     * Повернуть вокруг оси Z
     * @param angle Угол (в градусах)
     */
    RotateZ(angle: number);
    /**
     * Развернуть объект вдоль осей
     * @param axisz Ось Z
     * @param axisy Ось Y
     */
    Orient(axisz: Vector, axisy: Vector);
    /**
     * Развернуть объект вдоль глобальных осей
     * @param axisz
     * @param axisy
     */
    OrientGCS(axisz: Vector, axisy: Vector);
    /**
     * Преобразовать точку в ЛСК объекта
     * @param pos Координаты точки
     */
    ToObject(pos: Vector): Vector;
    /**
     * Преобразовать точку из ЛСК объекта
     * @param pos Координаты точки
     */
    ToGlobal(pos: Vector): Vector;
    /**
     * Преобразовать нормаль в ЛСК объекта
     * @param dir Координаты нормали
     */
    NToObject(dir: Vector): Vector;
    /**
     * Преобразовать нормаль из ЛСК объекта
     * @param dir Координаты нормали
     */
    NToGlobal(dir: Vector): Vector;
    /**
     * Локальные размеры объекта
     */
    GSize: Vector;
    /**
     * Габарит объекта в ЛСК
     */
    GMin: Vector;
    /**
     * Габарит объекта в ЛСК
     */
    GMax: Vector;
    /**
     * Габарит объекта
     */
    GabMin: Vector;
    /**
     * Габарит объекта
     */
    GabMax: Vector;
    /**
     * Получить список общего крепежа на двух панелях
     * @param Obj
     */
    FindConnectedFasteners(Obj?: Object3): Array<Object3>;
    /**
     * Получить список объектов, соединяемых этим крепежом
     */
    FindFastenedObjects(): Array<Object3>;
    /**
     * Количество пользовательских свойств
     */
    UserPropCount: number;
    /**
     * Значения свойства по его имени или индексу
     */
    UserProperty: Array<Object>;
    /**
     * Названия свойств
     */
    UserPropertyName: Array<string>;
    /**
     * Перестроить объект после изменения его свойств
     */
    Build();
    /**
     * Определить, является ли объект obj родителем объекта
     * @param obj Предполагаемый родитель объекта
     */
    IsOwner(obj: Object3): boolean;
    /**
     * Назначить локальные позицию и поворот из другого объекта
     * @param source
     */
    AssignTransform(source: Object3);
    /**
     * Идентификатор объекта
     */
    readonly UID: number;

    /** Проверка пересечения луча с объектом */
    RayIntersect(ray: RayInfo);
}

declare interface List3D extends Object3 {
    /**
     * Количество объектов в структуре
     */
    Count: number;
    /**
     * Получить элемент по индексу
     */
    [i: number]: Object3;
    /**
     * Список объектов
     */
    Objects: Array<Object3>;
    /**
     * Найти объект по имени
     * @param name Имя объекта
     */
    Find(name: string): Object3;
    /**
     * Являетсяли объект эластичным
     */
    IsElastic(): boolean;
    /**
     * Растянуть объект до требуемых размеров
     * @param newSize Новый размер
     */
    ElasticResize(newSize: Vector): Vector;
    /**
     * Загрузить объекты из файлов *.b3d,*.f3d
     * @param file Имя файла
     */
    Load(file: string): boolean;

}

declare interface Panel extends Object3 {
    /**
     * Контур панели
     */
    Contour: Contour2D;
    /**
     * Ширина контура панели
     */
    ContourWidth: number;
    /**
     * Высота контура панели
     */
    ContourHeight: number;
    /**
     * Толщина панели
     */
    Thickness: number;
    /**
     * Материал панели
     */
    MaterialName: string;
    /**
     * Ширина материала
     */
    MaterialWidth: number;
    /**
     * Ориентация текстуры
     */
    TextureOrientation: TextureOrientation;
    /**
     * Список кромок
     */
    Butts: PanelButts;
    /**
     * Список пластиков
     */
    Plastics: PanelPlastics;
    /**
     * Список пазов
     */
    Cuts: PanelCuts;
    /**
     * Закрыта ли кромка другими панелями? Указывается индекс кромки и расстояние до панелей.
     * @param index индекс кромки
     * @param distance расстояние до панелей
     */
    IsButtVisible(index: number, distance: number): boolean;
    /**
     * Является ли контур прямоугольным?
     */
    IsContourRectangle: boolean;
    /**
     * Накатать кромку на элемент
     * @param material Материал кромки
     * @param elem Элемент
     */
    AddButt(material: InMaterial | ScriptMaterialProperty, elem): PanelButt;
    /**
     * Наклеить пластик на панель
     * @param material Материал пластика
     * @param Front На лицевую пласть панели
     */
    AddPlastic(material: InMaterial | ScriptMaterialProperty, Front: boolean): PanelPlastic;
    /**
     * Создать новый паз
     * @param name
     */
    AddCut(name: string): PanelCut;

}

declare interface Extrusion extends Object3 {
    /**
     * Контур профиля
     */
    Contour: Contour2D;
    /**
     * Длина профиля
     */
    Thickness: number;
    /**
     * Материал
     */
    MaterialName: string;
    /**
     * Ширина материала
     */
    MaterialWidth: number;
    /**
     * Отрезать часть профиля в точке pos перпендикулярно normal
     * @param pos Координаты точки
     * @param normal Координаты нормали
     */
    Clip(pos: Vector, normal: Vector);

}

declare interface Trajectory extends Object3 {
    /**
     * Контур
     */
    Contour2D: Contour2D;
    /**
     * Траектория
     */
    Trajectory2D: Contour2D;
    /**
     * Наименование материала
     */
    MaterialName: string;
    /**
     * Ширина материала
     */
    MaterialWidth: number;

}

declare interface AnimBlock extends List3D {
    /**
     * Тип анимации
     */
    AnimType: AnimationType;
    /**
     * Угол открытия двери
     */
    DoorAngle: number;
    /**
     * Смещение двери
     */
    DoorShift: number;
}

declare interface Block extends AnimBlock {
    /**
     * Флаг составной фурнитуры
     */
    IsFastener(): boolean;

}

declare interface Assembly extends AnimBlock {
}

declare interface Contour3D extends Object3 {
    /**
     * Элементы вспомогательного контура
     */
    Contour: Contour2D;

}

declare interface Size3D extends Object3 {
    /**
     * Перестроить по точкам
     * @param Pos1
     * @param Pos2
     * @param TopPos
     */
    MakeOnPoints(Pos1, Pos2, TopPos);
    /**
     * Размер
     */
    Value: number;

}


/**
 * Типы анимации сборок и блоков
 */
declare enum AnimationType {
    /**
     * Не учитывается в салоне
     */
    None,
    /**
     * Учитывается в салоне, не имеет анимации
     */
    Custom,
    /**
     * Дверь левая
     */
    DoorLeft,
    /**
     * Дверь правая
     */
    DoorRight,
    /**
     * Дверь откидная
     */
    DoorFlap,
    /**
     * Дверь подъемная
     */
    DoorLift,
    /**
     * Дверь купе левая
     */
    SDoorLeft,
    /**
     * Дверь купе правая
     */
    SDoorRight,
    /**
     * Ящик
     */
    Box,
    /**
     * Опора
     */
    Support,
    /**
     * Ручка
     */
    Handle,
    /**
     * Фасад
     */
    Facade

}

/**
 * Направление текстуры материала на панели
 */
declare enum TextureOrientation {
    /**
     * Нет
     */
    None,
    /**
     * Горизонтально
     */
    Horizontal,
    /**
     * Вертикально
     */
    Vertical

}

declare interface PanelButts {
    /**
     * Добавить кромку
     */
    Add(): PanelButt;
    /**
     * Количество кромок
     */
    Count: number;
    /**
     * Список кромок
     */
    Butts: Array<PanelButt>;

}

declare interface PanelButt {
    /**
     * Индекс кромки
     */
    ElemIndex: number;
    /**
     * Условное обозначение
     */
    Sign: string;
    /**
     * Наименование материала
     */
    Material: string;
    /**
     * Толщина кромки
     */
    Thickness: number;
    /**
     * ширина
     */
    Width: number;
    /**
     * подрезать панель на толщину кромки
     */
    ClipPanel: boolean;
    /**
     * свес - насколько лента кромки должна быть длиннее торца детали
     */
    Overhung: number;
    /**
     * припуск - насколько прифрезеровать торец перед кромкованием
     */
    Allowance: number;
    /**
     * номер отреза (для наклейки кромки на несколько торцев одним отрезом)
     */
    CutIndex: number;
    /**
     * профиль кромки
     */
    Profile: Contour2D;

}

declare interface PanelPlastics {
    /**
     * Добавить пластик
     */
    Add(): PanelPlastic;
    /**
     * Количество пластика
     */
    Count: number;
    /**
     * Список
     */
    Plastics: Array<PanelPlastic>;

}

declare interface PanelPlastic {
    /**
     * Наименование материала
     */
    Material: string;
    /**
     * Толщина пластика
     */
    Thickness: number;
    /**
     * Ориентация текстуры
     */
    TextureOrientation: TextureOrientation;

}

declare interface PanelCuts {
    /**
     * Добавить паз
     */
    Add(): PanelCut;
    /**
     * Количество пазов
     */
    Count: number;
    /**
     * Список пазов
     */
    Cuts: Array<PanelCut>;

}

declare interface PanelCut {
    /**
     * Имя паза
     */
    Name: string;
    /**
     * Условное обозначение
     */
    Sign: string;
    /**
     * Траектория паза
     */
    Trajectory: Contour2D;
    /**
     * Профиль паза
     */
    Contour: Contour2D;

}
/**
 * Тип 2D элемента
 */
declare enum ElementType {
    /**
     * Неизвестный
     */
    Unknown,
    /**
     * Линия
     */
    Line,
    /**
     * Дуга
     */
    Arc,
    /**
     * Окружность
     */
    Circle,
    /**
     * Список элементов
     */
    List,
    /**
     * Эллипс
     */
    Ellipse
}
/**
 * 2D-элемент
 */
declare interface Elem2D {
    /**
     * Тип 2D элемента
     */
    ElType: ElementType;
    /**
     * Является ли элемент линией
     */
    IsLine(): boolean;
    /**
     * Получить элемент как линию
     */
    AsLine(): Line2D;
    /**
     * Является ли элемент окружностью
     */
    IsCircle(): boolean;
    /**
     * Получить элемент как окружность
     */
    AsCircle(): Circle2D;
    /**
     * Является ли элемент эллипсом
     */
    IsEllipse(): boolean;
    /**
     * Получить элемент как эллипс
     */
    AsEllipse(): Ellipse2D;
    /**
     * Является ли элемент дугой
     */
    IsArc(): boolean;
    /**
     * Получить элемент как дугу
     */
    AsArc(): Arc2D;
    /**
     * Является ли элемент списком элементов
     */
    IsList(): boolean;
    /**
     * Получить элемент как список элементов
     */
    AsList(): Contour2D;
    /**
     * Длина элемента
     */
    ObjLength(): number;
}
/**
 * 2D линия
 */
declare interface Line2D extends Elem2D {
    /**
     * Начало
     */
    Pos1: Point;
    /**
     * Конец
     */
    Pos2: Point;
}
/**
 * 2D дуга
 */
declare interface Arc2D extends Elem2D {
    /**
     * Начало
     */
    Pos1: Point;
    /**
     * Конец
     */
    Pos2: Point;
    /**
     * Центр
     */
    Center: Point;
    /**
     * Направление
     * True = против часовой стрелки
     * False = по часовой стрелке
     */
    ArcDir: boolean;

}
/**
 * 2D окружность
 */
declare interface Circle2D extends Elem2D {
    /**
     * Центр
     */
    Center: Point;
    /**
     * Радиус
     */
    CirRadius: number;
    /**
     * Направление
     * True = против часовой стрелки
     * False = по часовой стрелке
     */
    Dir: boolean;
}
/**
 * 2D эллипс
 */
declare interface Ellipse2D extends Elem2D {
    /**
     * Центр
     */
    Center: Point;
    /**
     * Большой радиус
     */
    MajorRadius: number;
    /**
     * Малый радиус
     */
    MinorRadius: number;
    /**
     * Угол
     */
    MajorAxisAngle: number;
    /**
     * Направление
     * True = против часовой стрелки
     * False = по часовой стрелке
     */
    Dir: boolean;
}

declare interface Contour2D {
    /**
     * Количество элементов контура
     */
    Count: number;
    /**
     * Список элементов контура
     */
    Objects: Elem2D[];
    /**
     * Элемент контура
     * @param index индекс элемента
     */
    [index: number]: Elem2D;
    /**
     * Ширина контура
     */
    Width: number;
    /**
     * Высота контура
     */
    Height: number;
    /**
     * Левый нижний угол охватывающего прямоугольника
     */
    Min: Point;
    /**
     * Правый верхний угол охватывающего прямоугольника
     */
    Max: Point;
    /**
     * Очистить контур
     */
    Clear();
    /**
     * Сдвинуть все элементы
     * @param dx Сдвиг по Х
     * @param dy Сдвиг по Y
     */
    Move(dx: number, dy: number);
    /**
     * Повернуть вокруг точки
     * @param x Координата Х
     * @param y Координата Y
     * @param angle Угол (в градусах)
     */
    Rotate(x: number, y: number, angle: number);
    /**
     * Добавить прямоугольник
     * @param x1
     * @param y1
     * @param x2
     * @param y2
     */
    AddRectangle(x1: number, y1: number, x2: number, y2: number);
    /**
     * Добавить прямоугольник со скурглёнными краями
     * @param x1
     * @param y1
     * @param x2
     * @param y2
     * @param rad
     */
    AddRoundRect(x1: number, y1: number, x2: number, y2: number, rad: number);
    /**
     * Добавить линию
     * @param x1
     * @param y1
     * @param x2
     * @param y2
     */
    AddLine(x1: number, y1: number, x2: number, y2: number): Object;
    /**
     * Добавить окружность
     * @param xc
     * @param yc
     * @param rad
     */
    AddCircle(xc: number, yc: number, rad: number): Object;
    /**
     * Добавить дугу по началу, концу и центру
     * @param p1
     * @param p2
     * @param centre
     * @param orient Ориентация против часовой стрелки
     */
    AddArc(p1: Point, p2: Point, centre: Point, orient: boolean): Object;
    /**
     * Добавить дугу по 3 точкам
     * @param p1
     * @param p2
     * @param p3
     */
    AddArc3(p1: Point, p2: Point, p3: Point): Object;
    /**
     * Добавить эквидистанту контура. Последние 2 параметра отвечают за направление и скругление
     * @param contour
     * @param offset
     * @param Side
     * @param Rounding
     * @param Pos
     */
    AddEquidistant(contour: Contour2D, offset: number, Side: boolean, Rounding: boolean, Pos?: Point);
    /**
     * Добавить эквидистанту контура (включая вложенные контуры). Последние 2 параметра отвечают за направление и скругление
     * @param contour
     * @param offset
     * @param Side
     * @param Rounding
     */
    AddEquidistantRecursive(contour: Contour2D, offset: number, Side: boolean, Rounding: boolean);
    /**
     * Вычесть замкнутый контур
     * @param contour
     */
    Subtraction(contour: Contour2D);
    /**
     * Сложить с замкнутым контуром
     * @param contour
     */
    Addition(contour: Contour2D);
    /**
     * Скругление элементов
     * @param elem1
     * @param elem2
     * @param x
     * @param y
     * @param radius
     */
    RoundingEx(elem1, elem2, x: number, y: number, radius): Object;
    /**
     * Фаска на 2 элементах
     * @param elem1
     * @param elem2
     * @param l1
     * @param l2
     */
    FacetEx(elem1, elem2, l1: number, l2?: number): Object;
    /**
     * Скругление в указанной точке
     * @param x
     * @param y
     * @param radius
     */
    Rounding(x: number, y: number, radius: number): Object;
    /**
     * Фаска в указанной точке
     * @param x
     * @param y
     * @param l
     */
    Facet(x: number, y: number, l: number): Object;
    /**
     * Найти ближайший элемент по координатам
     * @param x
     * @param y
     */
    Find(x: number, y: number): Object;
    /**
     * Вписать весь контур в заданные габариты
     * @param x1
     * @param y1
     * @param x2
     * @param y2
     */
    Fit(x1: number, y1: number, x2: number, y2: number);
    /**
     * Растянуть контур резиновой нитью
     * @param x1
     * @param y1
     * @param x2
     * @param y2
     * @param dx
     * @param dy
     */
    Elastic(x1: number, y1: number, x2: number, y2: number, dx: number, dy: number);
    /**
     * Отразить контур относительно линии
     * @param x1
     * @param y1
     * @param x2
     * @param y2
     * @param Copy
     */
    Symmetry(x1: number, y1: number, x2: number, y2: number, Copy: boolean);
    /**
     * Загрузить контур из файла *.frw
     * @param file Имя файла
     */
    Load(file: string): boolean;
    /**
     * Находится ли точка внутри контура?
     * @param x
     * @param y
     */
    IsPointInside(x: number, y: number): boolean;
    /**
     * Находится ли контур внутри другого?
     * @param contour
     */
    IsInContour(contour: Contour2D): boolean;
    /**
     * Является ли контур замкнутым?
     */
    IsClosedContour(): boolean;
    /**
     * Является ли контур прямоугольным?
     */
    IsContourRectangle(): boolean;
    /**
     * Упорядочить элеметны контура в одном направлении
     * @param closet
     */
    OrderContours(closet: boolean): boolean;

}

declare interface Geometry2D {
    /**
     * Вычислить точки пересечения двухмерных элементов
     * @param elem1
     * @param elem2
     */
    Intersect(elem1, elem2): Array<Point>;
    /**
     * Сравнить элементы
     * @param elem1
     * @param elem2
     */
    Compare(elem1, elem2): boolean;
    /**
     * Найти кратчайшее расстояние между элементами
     * @param elem1
     * @param elem2
     */
    Distance(elem1, elem2): number;
    /**
     * Найти площадь контура
     * @param contour
     */
    Area(contour: Contour2D): number;

}

declare interface VectorGeometry3D {
    /**
     * Нормализовать вектор
     * @param v
     */
    VectorNormalize(v: Vector): Vector;
    /**
     * Сложение двух векторов. Эквивалентно v1 + v2
     * @param v1
     * @param v2
     */
    VectorAdd(v1: Vector, v2: Vector): Vector;
    /**
     * Вычитание двух векторов. Эквивалентно v1 - v2
     * @param v1
     * @param v2
     */
    VectorSub(v1: Vector, v2: Vector): Vector;
    /**
     * Умножение вектора на число
     * @param v1
     * @param a
     */
    VectorMul(v1: Vector, a: number): Vector;
    /**
     * Деление вектора на число
     * @param v1
     * @param a
     */
    VectorDiv(v1: Vector, a: number): Vector;
    /**
     * Скалярное произведение векторов
     * @param v1
     * @param v2
     */
    VectorDot(v1: Vector, v2: Vector): number;
    /**
     * Является ли вектор пустым
     * @param v
     */
    VectorEmpty(v: Vector): boolean;
    /**
     * Векторное произведение векторов
     * @param v1
     * @param v2
     */
    VectorCross(v1: Vector, v2: Vector): Vector;
    /**
     * Создать копию точки
     * @param v
     */
    VectorCopy(v: Vector): Vector;
    /**
     * Вычислить длину вектора
     * @param v
     */
    VectorLength(v: Vector): number;
    /**
     * Инвертировать вектор
     * @param v
     */
    VectorNegate(v: Vector): Vector;
}

/**
 * Информация о простом соединении двух объектов
 */
declare interface ScriptSimpleConnection {
    /**
     * Установить фурнитуру на стык двух панелей
     * @param Furn Объект, содержащий информацию о фурнитуре
     * @param mountPoint Точка установки фурнитуры
     */
    Mount(Furn: InfFurniture | InFurniture, mountPoint: Vector): Object3;
    /**
     * Поменять начальную и конечную точки стыка
     */
    InvertPositions();
    /**
     * Изменить пласть установки (установить обратную сторону стыка)
     */
    ChangeSide();
    /**
     * Первая панель
     */
    Panel1: Panel;
    /**
     * Вторая панель
     */
    Panel2: Panel;
    /**
     * Начало стыка
     */
    PointStart: Vector;
    /**
     * Конец стыка
     */
    PointEnd: Vector;
    /**
     * Направление длины стыка (нормализованный вектор от начала до конца стыка)
     */
    Dir: Vector;
    /**
     * Направление ширины стыка (Нормализованный вектор от текущей до противоположной стороны стыка)
     */
    WidthDir: Vector;
}

/**
 * Информация об общем соединении двух объектов
 */
declare interface ScriptCommonConnection {
    /**
     * Получить элемент по индексу
     */
    [i: number]: ScriptSimpleConnection;
    /**
     * Получить элемент по индексу
     */
    Items: Array<ScriptSimpleConnection>;
    /**
     * Количество элементов
     */
    Count: number;
    /**
     * Первая панель, указанная при создании объекта
     */
    Panel1: Panel;
    /**
     * Вторая панель, указанная при создании объекта
     */
    Panel2: Panel;
}

/**
 * Элемент на динамической панели
 */
declare interface InControl {
    /**
     *
     */
    id: number;
    /**
     * Разрешить изменение свойства
     */
    Enabled: boolean;
    /**
     * Видимость свойства
     */
    Visible: boolean;
    /**
     * Подсказка
     */
    Hint: string;
    /**
     * Обработчик
     */
    OnChange: Function;

}

/**
 * Элемент "Кнопка" на динамической панели
 */
declare interface InButton extends InControl {
    /**
     * Создать подменю
     * @param caption
     */
    NewSubMenu(caption: string): InButton;
    /**
     *
     */
    Combo: boolean;

}

/**
 * Элемент ввода числа с плавающей точкой на динамической панели
 */
declare interface InFloat extends InControl {
    /**
     * Значение
     */
    Value: number;
    /**
     * Только для чтения
     */
    ReadOnly: boolean;
    /**
     *
     */
    Fixed: boolean;

}

/**
 * Элемент ввода целого числа на динамической панели
 */
declare interface InNumber extends InControl {
    /**
     * Значение
     */
    Value: number;
    /**
     * Только для чтения
     */
    ReadOnly: boolean;
    /**
     *
     */
    Fixed: boolean;

}

/**
 * Элемент выбора материала на динамической панели
 */
declare interface InMaterial extends InControl {
    /**
     * Наименование материала
     */
    Name: string;
    /**
     * Толщина материала
     */
    Thickness: number;
    /**
     * Ширина материала
     */
    Width: number;
    /**
     * Установить активным. Все последующие элементы будут построены из этого материала
     */
    SetActive();
    /**
     * Применить материал к указанному объекту
     * @param obj Объект
     */
    Apply(obj: Object3);

}
/**
 * Элемент выбора кромки на динамической панели
 */
declare interface InButtMaterial extends InControl {
    /**
     * Наименование кромки
     */
    Name: string;
    /**
     * Условное обозначение
     */
    Sign: string;
    /**
     * Толщина кромки
     */
    Thickness: number;
    /**
     * Ширина кромки
     */
    Width: number;
    /**
     * подрезать панель на толщину кромки
     */
    ClipPanel: boolean;
    /**
     * свес - насколько лента кромки должна быть длиннее торца детали
     */
    Overhung: number;
    /**
     * припуск - насколько прифрезеровать торец перед кромкованием
     */
    Allowance: number;

}
/**
 * Элемент выбора фурнитуры на динамической панели
 */
declare interface InFurniture extends InControl {
    /**
     * Установить крепеж между двух панелей
     * @param panel1
     * @param panel2
     * @param x
     * @param y
     * @param z
     */
    Mount(panel1: Panel, panel2: Panel, x: number, y: number, z: number): Object3;
    /**
     * Установить крепеж на плоскость панели
     * @param panel
     * @param x
     * @param y
     * @param z
     * @param angle Угол (в градусах)
     */
    Mount1(panel: Panel, x: number, y: number, z: number, angle: number): Object3;
    /**
     * Установить схему крепежа на стык панелей
     * @param panel1
     * @param panel2
     * @param FurniturePosition позиция фурнитуры
     * @param BasisPoint Базовая точка
     */
    MountScheme(panel1: Panel, panel2: Panel, FurniturePosition: FurniturePosition, BasisPoint: Vector): Object3;
    /**
     * Установить секцию
     * @param Position Позиция секции
     * @param Size Размер секции
     * @param axisZ Направление оси Z секции
     * @param axisY Направление оси Y секции
     */
    MountBox(Position: Vector, Size: Vector, axisZ: Vector, axisY: Vector): Object3;
    /**
     * Фильтр по типу монтирования фурнитуры/фрагмента
     */
    DatumModeFilter: DatumMode;
    /**
     * Тип монтирования выбранной фурнитуры/фрагмента
     */
    DatumMode: DatumMode;
}

/**
 * Информация о фурнитуре
 */
declare interface InfFurniture {
    /**
     * Установить крепеж между двух панелей
     * @param panel1
     * @param panel2
     * @param x
     * @param y
     * @param z
     */
    Mount(panel1: Panel, panel2: Panel, x: number, y: number, z: number): Object3;
    /**
     * Установить крепеж на плоскость панели
     * @param panel
     * @param x
     * @param y
     * @param z
     * @param angle Угол (В градусах)
     */
    Mount1(panel: Panel, x: number, y: number, z: number, angle: number): Object3;
    /**
     * Установить схему крепежа на стык панелей
     * @param panel1
     * @param panel2
     * @param FurniturePosition позиция фурнитуры
     * @param BasisPoint Базовая точка
     */
    MountScheme(panel1: Panel, panel2: Panel, FurniturePosition: FurniturePosition, BasisPoint: Vector): Object3;
    /**
     * Установить секцию
     * @param Position Позиция секции
     * @param Size Размер секции
     * @param axisZ Направление оси Z секции
     * @param axisY Направление оси Y секции
     */
    MountBox(Position: Vector, Size: Vector, axisZ: Vector, axisY: Vector): Object3;
    /**
     * Создать объект фурнитуры
     */
    Make(Thickness1: number, Thickness2: number): Object3;
    /**
     * Фильтр по типу монтирования фурнитуры/фрагмента
     */
    DatumModeFilter: DatumMode;
    /**
     * Тип монтирования выбранной фурнитуры/фрагмента
     */
    DatumMode: DatumMode;
    /**
     * Зашифровать параметры фурнитуры в строку
     */
    EncodeToString(): string;
    /**
     * Восстановить параметры фурнитуры из строки
     */
    DecodeFromString(str: string);
    /**
     * Открыть окно выбора фурнитуры (возвращает false, если выбор отменен)
     */
    Choose(): boolean;
}

declare interface DoorsMaker {
    /**
     *
     */
    Silent: boolean;
    /**
     *
     */
    ShowErrors: boolean;
    /**
     * Сохранить параметры установки дверей в файл
     * @param filename
     */
    Save(filename: string);
    /**
     * Загрузить параметры установки дверей из файла
     * @param filename
     */
    Load(filename: string): boolean;
    /**
     * Установить двери в секцию (Объект - Panel или Edge)
     * @param LeftObject
     * @param RightObject
     * @param TopObject
     * @param BottomObject
     */
    Setup(LeftObject: Panel | Edge3, RightObject: Panel | Edge3, TopObject: Panel | Edge3, BottomObject: Panel | Edge3);

}

declare interface BoxesMaker {
    /**
     *
     */
    ShowErrors: boolean;
    /**
     * Сохранить параметры установки ящиков в файл
     * @param filename
     */
    Save(filename: string);
    /**
     * Загрузить параметры установки ящиков из файла
     * @param filename
     */
    Load(filename: string): boolean;
    /**
     * Установить ящик в секцию. LeftObject, RightObject - панели. TopObject, BottomObject - Panel или Edge
     * @param LeftObject Левая панель
     * @param RightObject Правая панель
     * @param TopObject Верхняя граница (Панель или ребро)
     * @param BottomObject Нижняя граница (Панель или ребро)
     */
    Setup(LeftObject: Panel, RightObject: Panel, TopObject: Panel | Edge3, BottomObject: Panel | Edge3);

}

declare interface ImportExport {
    /**
     * Импорт-экспорт в SVG
     */
    SVG: ImportExportSVG;
}

declare interface ImportExportSVG {
    /**
     * Качество аппроксимации кривых в линии [0-100]
     */
    CurveQuality: number;
    /**
     * Объединять элементы в полигоны, полилинии и кривые
     */
    GroupElems: boolean;
    /**
     * Сохранить контур
     * @param filename Имя файла.
     * @param contour Контур, данные которого сохранятся в файл
     */
    Save(filename: string, contour: Contour2D): boolean;
    /**
     * Загрузить контур
     * @param filename Имя файла.
     * @param contour Контур, куда считаются данные из файла
     */
    Load(filename: string, contour: Contour2D): boolean;
}

/**
 * Элемент товара
 */
declare interface ScItemTovar {
    /**
     * Артикул элемента товара
     */
    Article: string;
    /**
     * Имя элемента товара
     */
    Name: string;
    /**
     * Текущий материал
     */
    Material: string;
    /**
     * Имя группы материалов на замену
     */
    GroupMaterial: string;
    /**
     * Имя типа элемента
     */
    TypeElement: string;
    /**
     * Список объектов из модели входящих в состав элемента товара
     */
    ObjList: List3D;

}

/**
 * Товар
 */
declare interface ScItemTovarList {
    /**
     * Доступ к элементу товара по индексу
     * @param [index]
     */
    Items: Array<ScItemTovar>;
    /**
     * Количество элементов товара
     */
    Count: number;
    /**
     * Имя товара
     */
    TovarName: string;
    /**
     * Артикул товара
     */
    TovarArticul: string;
    /**
     * Найти элемент товара по имени
     * @param name
     * @param CaseSensitive Учитывать регистр
     */
    FindByName(name: string, CaseSensitive: boolean): ScItemTovar;
    /**
     * Является ли объект нестандартным
     */
    IsNotStandart: boolean;

}

/**
 * Доп. функции
 */
declare interface TSalonUtils {
    /**
     * Путь к папке с прикрепленными файлами
     */
    PathAttachments: string;

    /**
     * Получить полное имя прикрепленного файла
     * @param fileName Имя файла
     */
    GetFullPathAttachment(fileName: string): string;

}

declare interface Arguments extends Object {
    /**
     *
     */
    callee: Function;
    /**
     * Количество аргументов
     */
    length: number;

}
/**
 * Создать 3D точку по координатам
 * @param x
 * @param y
 * @param z
 */
declare function NewVector(x: number, y: number, z: number): Vector;

/**
 * Создать 2D точку по координатам
 * @param x
 * @param y
 */
declare function NewPoint(x: number, y: number): Point;

/**
 * Создать плоский контур
 */
declare function NewContour(): Contour2D;

/**
 * Создать луч для проверки пересечения с объектами
 */
declare function NewRayInfo(): RayInfo;

/**
 * Создать новый COM объект по его типу
 * @param CLSID
 */
declare function NewCOMObject(CLSID: string): IDispatch;

/**
 * Создать форму со свойствами
 * @param caption Заголовок формы
 */
declare function NewForm(caption: string): ScriptForm;

/**
 * Создать базу параметрического крепежа
 */
declare function NewParamFastenerDB(): ScriptParamFastenerDB;

declare function NewPanelsConnection(p1: Panel, p2: Panel): ScriptCommonConnection;

    /**
     * Нормаль X
     */
    declare var AxisX: Vector;

/**
 * Нормаль Y
 */
declare var AxisY: Vector;

/**
 * Нормаль Z
 */
declare var AxisZ: Vector;

/**
 * Нормаль -X
 */
declare var Axis_X: Vector;

/**
 * Нормаль -Y
 */
declare var Axis_Y: Vector;

/**
 * Нормаль -Z
 */
declare var Axis_Z: Vector;

/**
 * Системные функции
 */
declare var system: System;

/**
 * Структура модели
 */
declare var Model: Model3D;

/**
 * Вспомогательные геометрические функции
 */
declare var geometry: Geometry2D;

/**
 * Вспомогательные функции для работы с векторами
 */
declare var vectorGeometry: VectorGeometry3D;

/**
 * Активный скрипт
 */
declare var Action: Action3D;

/**
 * Артикул модели
 */
declare var Article: FurnArticle;

/**
 * Текущий материал
 */
declare var ActiveMaterial: FurnMaterial;

/**
 * История модели
 */
declare var Undo: Undo3D;

/**
 * Вывести окно ввода строки
 * @param message Текст сообщения
 */
declare function prompt(message: string): string;

/**
 * Вывести окно сообщения
 * @param str Текст сообщения
 */
declare function alert(str: string);

/**
 * Показать окно подтверждения (Да/Нет)
 * @param message Текст сообщения
 */
declare function confirm(message: string): boolean;

/**
 * Открыть фурнитуру для установки на модель
 * @param filename Имя файла
 */
declare function OpenFurniture(filename: string): InfFurniture;

/**
 * Выделить всё
 */
declare function SelectAll();

/**
 * Снять выделение с модели
 */
declare function UnSelectAll();

/**
 * Показать всё
 */
declare function ViewAll();

/**
 * Установить текущий вид
 * @param p3d
 */
declare function SetCamera(p3d: Vector);

/**
 * Запрос точки
 * @param hint Текст подсказки
 */
declare function GetPoint(hint: string): Vector;

/**
 * Запрос объекта модели
 * @param hint Текст подсказки
 */
declare function GetObject(hint: string): Object3;

/**
 * Запрос панели
 * @param hint Текст подсказки
 */
declare function GetPanel(hint: string): Panel;

/**
 * Запрос выбора ребра, параллельного указанному вектору
 * @param hint Текст подсказки
 * @param Axis
 */
declare function GetEdge(hint: string, Axis: Vector): Edge3;

/**
 * Создать панель указанных размеров
 * @param width Ширина
 * @param height Высота
 */
declare function AddPanel(width: number, height: number): Panel;

/**
 * Создать фронтальную панель
 * @param x1
 * @param y1
 * @param x2
 * @param y2
 * @param z
 */
declare function AddFrontPanel(x1: number, y1: number, x2: number, y2: number, z: number): Panel;

/**
 * Создать горизонтальную панель
 * @param x1
 * @param z1
 * @param x2
 * @param z2
 * @param y
 */
declare function AddHorizPanel(x1: number, z1: number, x2: number, z2: number, y: number): Panel;

/**
 * Создать вертикальную панель
 * @param z1
 * @param y1
 * @param z2
 * @param y2
 * @param x
 */
declare function AddVertPanel(z1: number, y1: number, z2: number, y2: number, x: number): Panel;

/**
 * Создать профиль
 * @param name Имя объекта
 */
declare function AddExtrusion(name: string): Extrusion;

/**
 * Создать тело по траектории
 * @param name Имя объекта
 */
declare function AddTrajectory(name: string): Trajectory;

/**
 * Создать мебельный блок
 * @param name Имя объекта
 */
declare function AddBlock(name: string): Block;

/**
 * Создать полуфабрикат
 * @param name Имя объекта
 */
declare function AddDraftBlock(name: string): Block;

/**
 * Создать мебельную сборку
 * @param name Имя объекта
 */
declare function AddAssembly(name: string): Block;

/**
 * Создать копию объекта
 * @param obj
 */
declare function AddCopy(obj: Object3): Object3;

/**
 * Создать симметричную копию объекта
 * @param obj
 * @param  pos
 * @param  normal
 */
declare function AddSymmetry(obj: string, pos: Vector, normal: Vector): Object3;

/**
 * Создать вспомогательный контур в пространстве
 */
declare function AddContour(): Contour3D;

/**
 * Создать размер
 * @param pos1
 * @param pos2
 * @param toppos
 */
declare function AddSize(pos1: Vector, pos2: Vector, toppos: Vector): Size3D;

/**
 * Удалить объекты ранее созданные в скрипте
 */
declare function DeleteNewObjects();

/**
 * Удалить объект из модели
 * @param obj
 */
declare function DeleteObject(obj: Object3);

/**
 * Функция начала редактирования объекта
 * @param obj
 */
declare function StartEditing(obj: Object3): Object3;

/**
 * Начать создание блока. Все созданные далее объекты попадают внутрь блока
 * @param name Имя объекта
 */
declare function BeginBlock(name: string): Block;

/**
 * Закончить создание блока
 */
declare function EndBlock();

/**
 * Начать создание редактируемого блока
 * @param name Имя объекта
 */
declare function BeginParametricBlock(name: string): Block;

/**
 * Закончить создание редактируемого блока
 */
declare function EndParametricBlock();

/**
 * Редактируемый блок. Переменная установлена, если скрипт запущен в режиме редактирования
 */
declare var ParametricBlock: Block;

/**
 * Создать элемент управления - кнопку
 * @param caption
 */
declare function NewButtonInput(caption: string): InButton;

/**
 * Создать элемент управления для ввода целого числа
 * @param caption
 */
declare function NewFloatInput(caption: string): InFloat;

/**
 * Создать элемент управления для ввода числа
 * @param caption
 */
declare function NewNumberInput(caption: string): InNumber;

/**
 * Создать элемент управления для выбора материала
 * @param caption
 */
declare function NewMaterialInput(caption: string): InMaterial;

/**
 * Создать элемент управления для выбора кромочного материала
 * @param caption
 */
declare function NewButtMaterialInput(caption: string): InButtMaterial;

/**
 * Создать элемент управления для выбора фурнитуры
 * @param caption
 */
declare function NewFurnitureInput(caption: string): InFurniture;

/**
 * Создать элемент управления для анализа модели
 */
declare function NewModelInspector(): ModelInspector;

/**
 * Создать мастер установки дверей
 * @param caption
 */
declare function NewDoorsMaker(caption: string): DoorsMaker;

/**
 * Создать мастер установки ящиков
 * @param caption
 */
declare function NewBoxesMaker(caption: string): BoxesMaker;

/**
 * Создать объект для импорта-экспорта
 */
declare function NewImportExport(): ImportExport;

/**
 * Форматировать имя материала
 */
declare function FormatMatName(matName: string): string;

/**
 * Повернуть камеру
 * @param dir Новое направление взгляда
 */
declare function OrientCamera(dir: Vector);
/**
 * Извлечь имя материала.
 * @param material Полное имя материала
 */
declare function ExtractMatName(material: string): string;
/**
 * Извлечь код (артикул) материала.
 * @param material Полное имя материала
 */
declare function ExtractMatCode(material: string): string;

/**
 * Загрузить модель из файла
 */
declare function LoadModel(filename: string): boolean;
/**
 * Сохранить модель в файл
 */
declare function SaveModel(filename: string);
/**
 * Создать новую модель
 */
declare function NewModel();
/**
 * Создать новую фурнитуру
 */
declare function NewFurniture();
/**
 * Создать новый фрагмент
 */
declare function NewFragment();
/**
 * Имя файла текущей модели
 */
declare var ModelFilename: string;

/**
 * Список элементов товара. Только для Салона
 */
declare var TovarItems: ScItemTovarList;

/**
 * Вспомогательные функции для работы с Салоном
 */
declare var SalonUtils: TSalonUtils;